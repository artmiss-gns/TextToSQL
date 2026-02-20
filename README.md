# Text-to-SQL Fine-Tuning with Flan-T5-Large + LoRA

Fine-tuning `google/flan-t5-large` on the [Spider](https://huggingface.co/datasets/xlangai/spider) benchmark dataset to translate natural language questions into SQL queries, using parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation).

**Result: 22.4% exact match and 80% ROUGE-1 score on Spider validation set** — achieved with a 780M parameter model, LoRA adapters (~1.9% trainable parameters), and 3 hours of training on a single L4 GPU.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture & LoRA](#model-architecture--lora)
- [Training Details](#training-details)
- [Results](#results)
- [Debugging Journey](#debugging-journey)
- [Error Analysis](#error-analysis)
- [How to Use](#how-to-use)
- [Future Work](#future-work)

---

## Project Overview

Text-to-SQL is the task of translating a natural language question (e.g., *"How many players are from each country?"*) into an executable SQL query (`SELECT count(*), country_code FROM players GROUP BY country_code`), given a database schema as context.

This is a non-trivial generation task because the model must:
- Understand the semantics of the question
- Ground the output to the specific schema (table names, column names, relationships)
- Produce syntactically valid SQL with correct clause ordering

Spider is a widely used cross-domain, cross-database benchmark for this task. It contains ~7,000 training examples spanning 200+ databases, making generalization the key challenge.

The goal of this project was to fine-tune a mid-size seq2seq model (flan-t5-large) with LoRA to achieve competitive accuracy within tight compute constraints, and to document the engineering decisions and debugging process along the way.

---

## Dataset

**Sources:**
- [`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider) — main Spider dataset (train + validation splits)
- [`richardr1126/spider-schema`](https://huggingface.co/datasets/richardr1126/spider-schema) — pre-extracted schema strings per `db_id`

**Input format:**

Each training example is formatted as:

```
Translate English to SQL: <question> | Schemas: <schema>
```

Where `<schema>` is cleaned and formatted as:

```
table1(col1, col2, col3) | table2(col1, col2)
```

The raw schema strings from `richardr1126/spider-schema` have the form:
```
table : col1 (type) , col2 (type) | table2 : col3 (type)
```

A preprocessing function strips the type annotations and reformats into the compact `table(col1, col2)` style.

**Preprocessing steps:**
1. Remove unused columns (`query_toks`, `query_toks_no_value`, `question_toks`)
2. Build a `db_id → schema` lookup dictionary from the schema dataset
3. Apply tokenization with `remove_columns` to strip raw string columns from the dataset — this is critical; leaving string columns in causes `DataCollatorForSeq2Seq` to silently corrupt batches
4. Split the validation set 50/50 into validation and test

**Label handling:**

Labels use `-100` as the padding token ID (via `DataCollatorForSeq2Seq`), which is automatically ignored by PyTorch cross-entropy loss. This is distinct from input padding which uses `attention_mask`.

---

## Model Architecture & LoRA

**Base model:** `google/flan-t5-large` (780M parameters)

Flan-T5 is an encoder-decoder (seq2seq) transformer, instruction-tuned on a large mixture of tasks. It uses relative positional embeddings and is well-suited for structured generation tasks.

**Why flan-t5-large over base (250M)?**

| Model | Exact Match (epoch 8) | Training Time |
|---|---|---|
| flan-t5-base (full fine-tune) | 5.8% | ~45 min / 10 epochs |
| flan-t5-large + LoRA | **22.4%** | ~3 hours / 10 epochs |

Model capacity matters significantly for Spider. The base model plateaued early — it lacked the capacity to memorize SQL patterns while simultaneously learning schema grounding across 200+ databases.

**LoRA configuration:**

```python
LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
    lora_dropout=0.0,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM,
)
```

Trainable parameters: ~14M out of 780M (~1.9%)

**Critical: why target FFN layers (`wi_0`, `wi_1`, `wo`)?**

Most LoRA tutorials target only attention layers (`q`, `v`). For this task, that is insufficient. FFN layers are responsible for storing learned patterns — for SQL generation, this means clause structures, keyword sequences, and join patterns. Restricting LoRA to attention layers on a 250M model (flan-t5-base) produced nearly no improvement (loss moved from 9.7 → 6.6 in 100 steps). Including FFN layers in the adapter targets was essential.

**Mixed precision dtype strategy:**

A key engineering decision: the base model weights are stored in `bfloat16` (memory savings), while LoRA adapter weights are kept in `float32` (gradient stability).

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)   # LoRA adapters
    else:
        param.data = param.data.to(torch.bfloat16)  # frozen base weights
```

This is applied manually rather than using `bf16=True` in `TrainingArguments`, because the Trainer's `bf16` flag casts everything — including LoRA weights — to bf16. Small gradients in bf16 can underflow to zero, causing LoRA adapters to stop learning entirely. See the [Debugging Journey](#debugging-journey) for the full story.

**T5-specific decoder configuration:**

Unlike most decoder-based models, T5 uses the pad token (id=0) as the `decoder_start_token_id`, not a `<bos>` token. This must be set explicitly:

```python
model.config.decoder_start_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
```

---

## Training Details

**Hardware:** Single NVIDIA L4 GPU (22GB VRAM), Google Colab Pro

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.1 |
| Optimizer | AdamW (torch) |
| Per-device batch size | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Epochs | 10 |
| Weight decay | 5e-4 |
| Max grad norm | 2.0 |
| Label smoothing | 0 (disabled) |
| bf16 | False (manual mixed precision instead) |

**Notable decisions:**

- **Label smoothing disabled:** Inflates training loss and empirically hurt performance on this task
- **`repetition_penalty` removed:** SQL naturally repeats table and column names; a repetition penalty causes hallucinations in structured output
- **`group_by_length=True`:** Groups similarly-lengthed sequences together to minimize padding waste per batch
- **`load_best_model_at_end=True`:** Saves the checkpoint with the best `exact_match` on the validation set; the model peaked at epoch 8

**Evaluation metric:**

Exact match after normalizing both prediction and gold label (lowercased, whitespace stripped):

```python
def normalize_sql(s):
    return s.lower().replace(" ", "").strip()
```

ROUGE-1 is also tracked as a complementary signal for partial matches.

---

## Results

### Training Curve

| Epoch | Train Loss | Val Loss | Exact Match | ROUGE-1 |
|---|---|---|---|---|
| 1 | 11.822 | 3.050 | 0.19% | 30.78% |
| 2 | 7.482 | 2.201 | 4.25% | 55.00% |
| 3 | 6.382 | 1.834 | 11.41% | 64.78% |
| 4 | 5.817 | 1.746 | 13.54% | 68.13% |
| 5 | 4.988 | 1.602 | 14.89% | 73.07% |
| 6 | 5.238 | 1.544 | 19.54% | 77.67% |
| 7 | 4.573 | 1.554 | 21.28% | 78.91% |
| **8** | **4.326** | **1.542** | **22.44%** | **79.61%** |
| 9 | 4.576 | 1.538 | 22.05% | 79.98% |
| 10 | 4.466 | 1.442 | 21.86% | 79.88% |

**Best checkpoint: Epoch 8** (saved by `load_best_model_at_end=True`)

The model peaked at epoch 8 and showed a slight plateau/regression afterward, typical of cosine scheduling with a fixed epoch budget.

### Qualitative Example (debug sample, same question each epoch)

**Question:** *How many players are from each country?*
**Gold:** `SELECT count(*) , country_code FROM players GROUP BY country_code`

| Epoch | Prediction |
|---|---|
| 1 | `SELECT count(*) FROM player_id` |
| 2 | `SELECT count(*) , country_code FROM players` |
| 3 | `SELECT count(*) , country_code FROM players WHERE count(*) = 1` |
| 4 | `SELECT count(*) , country_code FROM athletes WHERE count(*) = (SELECT ...)` |
| 5–10 | `SELECT count(*) , country_code FROM players GROUP BY country_code` ✅ |

The model learned the correct structure progressively: first the SELECT columns, then the correct table, then GROUP BY.

---

## Debugging Journey

This section documents the non-obvious bugs encountered during development — the kind that don't appear in tutorials.

### 1. Dtype Mismatch (Primary Training Failure)

**Symptom:** Training loss stuck at 22–24 and not decreasing. Generated output was complete nonsense.

**Root cause:** When loading a base model in `bfloat16` and applying LoRA with `get_peft_model`, the adapter weights initialize in `float32`. At each forward pass, the input (bf16) is multiplied by lora_A (fp32) — PyTorch implicitly upcasts, but the gradients flowing back to the base model cause silent numerical issues that compound. The mismatch between base and adapter precision made training unstable.

**Fix:** Load the base model in the default dtype (fp32), apply LoRA, then manually cast:
- Frozen base weights → `bfloat16` (memory efficiency)
- LoRA adapter weights → `float32` (gradient precision)

Do **not** use `bf16=True` in `TrainingArguments` — this casts everything including adapters to bf16 during the forward pass, which causes the same problem.

### 2. LoRA Gradient Flow (Expected Behavior, Mistaken for a Bug)

**Symptom:** At step 0, all `lora_A` weights had zero gradients. Only `lora_B` had non-zero gradients.

**Explanation:** This is mathematically expected. Standard LoRA initialization sets `lora_B = 0`, so the output of the adapter at initialization is `lora_A @ lora_B = 0`. The gradient to `lora_A` is `∂loss/∂output × lora_B^T = 0`. After the first optimizer step, `lora_B` is no longer zero, and `lora_A` begins receiving gradients.

**Verified:** Gradient norms at lora_A across steps: `0.0 → 0.0004 → 0.0012 → 0.0027`. Not a bug.

### 3. `remove_columns` Missing from Dataset Preprocessing

**Symptom:** Training appeared to run but metrics were garbage.

**Root cause:** The `remove_columns=dataset['train'].column_names` argument was commented out in the `.map()` call. This left raw string columns (`question`, `query`, `db_id`) in the dataset alongside the tokenized columns. `DataCollatorForSeq2Seq` cannot batch strings and silently corrupted the batch construction.

**Fix:** Always include `remove_columns` when tokenizing:

```python
dataset['train'] = dataset['train'].map(
    preprocess,
    batched=True,
    remove_columns=dataset['train'].column_names,  # critical
)
```


### 4. Out-of-Range Token IDs in Predictions

**Symptom:** Occasional crashes in `compute_metrics` with index errors.

**Root cause:** In bf16 training, occasional numerical issues produce token IDs outside the vocabulary range, causing `tokenizer.decode` to crash.

**Fix:** Clip predictions before decoding:

```python
preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
```

---

## Error Analysis

Testing on 12 samples from the test set revealed consistent failure patterns:

**Strengths (model handles well):**
- Single-table SELECT with filtering (`WHERE`, `ORDER BY`, `LIMIT`)
- Simple aggregations (`COUNT`, `AVG`, `MAX`)
- Single JOIN with clear column references

**Failure modes:**

| Pattern | Example | Notes |
|---|---|---|
| `EXCEPT` / `UNION` / `INTERSECT` | *"Which airlines depart from CVO but not APG?"* | Model never generates set operations; uses `WHERE` as a workaround instead |
| 3+ table JOINs | *"What are the names of people who teach math courses?"* | Picks correct columns but wrong intermediate join table |
| `HAVING` clause | *"Which departments have more than 5 employees?"* | Confuses `HAVING` with `WHERE`; filters in the wrong clause |
| Repetition loops | `GROUP BY x GROUP BY x GROUP BY x` | Rare but occurs; model loops on structured tokens |

All of these failure modes stem from insufficient training examples of those specific patterns. Spider's 7k examples provide limited coverage of complex set operations and deep multi-join queries.

**Note on exact match as a metric:** String exact match undercounts true accuracy. Queries like `SELECT count(*), country_code FROM players GROUP BY country_code` vs `SELECT country_code, count(*) FROM players GROUP BY country_code` are semantically equivalent but counted as wrong. Execution-based matching (running both queries and comparing result sets) is the proper evaluation methodology for Spider.

---

## How to Use

**Installation:**

```bash
pip install transformers peft torch
```

**Load the model:**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("artmiss/flan-t5-large-spider-text2sql")
model = PeftModel.from_pretrained(base_model, "artmiss/flan-t5-large-spider-text2sql")
model.eval()
```

> **Note:** The HuggingFace repo contains only the LoRA adapter weights (~50MB). The base model must be loaded separately from `google/flan-t5-large` before applying the adapter.

**Run inference:**

```python
def predict(question: str, schema: str) -> str:
    """
    Args:
        question: Natural language question
        schema: Schema string in the format "table1(col1, col2) | table2(col3)"
    Returns:
        SQL query string
    """
    input_text = f"Translate English to SQL: {question} | Schemas: {schema}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Example:**

```python
schema = "players(player_id, first_name, last_name, country_code) | matches(match_id, winner_id, loser_id)"
question = "How many players are from each country?"

print(predict(question, schema))
# → SELECT count(*) , country_code FROM players GROUP BY country_code
```

---

## Future Work

- **Larger training data:** Adding [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (~78k examples) would provide significantly more coverage of complex SQL patterns (EXCEPT, UNION, 3+ table joins, HAVING)
- **Execution-based evaluation:** Replace string exact match with execution accuracy using the official Spider evaluation script
- **Schema linking:** Add a pre-processing step to identify relevant tables/columns before generation, reducing the context complexity and improving JOIN accuracy
- **Larger model:** `flan-t5-xl` (3B parameters) with the same LoRA setup would likely push accuracy into the 30–35% range

---

## References

- [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Flan-T5: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)