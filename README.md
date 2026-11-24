# Text-to-SQL Fine-Tuning with Flan-T5-Large + LoRA

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-brightgreen)](https://huggingface.co/spaces/artmiss/text-to-sql)
[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-blue)](https://huggingface.co/artmiss/flan-t5-large-spider-text2sql)
[![Dataset](https://img.shields.io/badge/Dataset-Spider-orange)](https://huggingface.co/datasets/xlangai/spider)

Fine-tuning `google/flan-t5-large` on the [Spider](https://huggingface.co/datasets/xlangai/spider) benchmark dataset to translate natural language questions into SQL queries, using parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation).

**Result: 22.44% normalized SQL string exact match and 79.61% ROUGE-1 at epoch 8 on a 517-example subset of Spider’s validation split.** This subset was also used to select the best checkpoint. The run took about 3 hours on a single L4 GPU.

---

## Live Demo

**Try it in your browser — no setup required:**
👉 [huggingface.co/spaces/artmiss/text-to-sql](https://huggingface.co/spaces/artmiss/text-to-sql)

Build a schema by adding tables and columns one by one, type a natural language question, and get a syntax-highlighted SQL query back instantly.

> **Note:** The demo runs on a free CPU instance. First load after inactivity may take ~30 seconds to wake up. Inference takes ~10–20 seconds per query.

---

## Table of Contents

- [Live Demo](#live-demo)
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

Spider is a widely used cross-domain, cross-database benchmark for this task. This project uses its 7,000 training examples, with generalization to new schemas as the main challenge.

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
3. Tokenize with `remove_columns` so the resulting dataset contains only `input_ids`, `attention_mask`, and `labels`
4. Split Spider’s validation set 50/50 into a 517-example validation subset and a 517-example holdout using seed 42

**Label handling:**

Labels use `-100` as the padding token ID (via `DataCollatorForSeq2Seq`), which is automatically ignored by PyTorch cross-entropy loss. This is distinct from input padding which uses `attention_mask`.

---

## Model Architecture & LoRA

**Base model:** `google/flan-t5-large` (780M parameters)

Flan-T5 is an encoder-decoder (seq2seq) transformer, instruction-tuned on a large mixture of tasks. It uses relative position biases in attention and is well-suited for structured generation tasks.

**Why flan-t5-large over base (250M)?**

| Model | Exact Match (epoch 8) | Training Time |
|---|---|---|
| flan-t5-base (full fine-tune) | 5.8% | ~45 min / 10 epochs |
| flan-t5-large + LoRA | **22.4%** | ~3 hours / 10 epochs |

The base model plateaued earlier in my experiments. These runs used different model sizes and fine-tuning methods, so the comparison cannot isolate the effect of model capacity.

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

The base model is frozen during LoRA fine-tuning. The Colab export prints the trainable parameter count after attaching the adapters.

**Critical: why target FFN layers (`wi_0`, `wi_1`, `wo`)?**

In a short attention-only LoRA experiment on FLAN-T5-Base, the loss went from 9.7 to 6.6 over 100 steps. I also targeted the feed-forward layers in the large-model setup to give the adapters more scope to learn SQL patterns. The two experiments do not isolate the effect of adding feed-forward targets.

**Mixed precision dtype strategy:**

A key engineering decision: the base model weights are stored in `bfloat16` (memory savings), while LoRA adapter weights are kept in `float32` (gradient stability).

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)   # LoRA adapters
    else:
        param.data = param.data.to(torch.bfloat16)  # frozen base weights
```

The manual cast is active in `texttosql.py` and commented out in `flan_t5_large_spider_text2sql.ipynb`. Both files set `bf16=True` in `TrainingArguments` to enable BF16 mixed-precision computation. See the [Debugging Journey](#debugging-journey) for the context behind the manual cast.

**T5-specific decoder configuration:**

T5 uses the pad token (ID 0) as its `decoder_start_token_id`. The training code sets the decoder start, end-of-sequence, and pad token IDs explicitly:

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
| bf16 | True in both code versions. `texttosql.py` also casts frozen base weights to BF16 and trainable LoRA weights to FP32. The exact precision configuration of the full reported run has not been verified. |

**Notable decisions:**

- **Label smoothing disabled:** The training configuration uses `label_smoothing_factor=0`
- **`repetition_penalty` omitted from evaluation generation:** SQL often repeats table and column names. The notebook’s debug callback still uses `1.2`. The callback in `texttosql.py` omits it.
- **`group_by_length=True`:** Groups similarly-lengthed sequences together to minimize padding waste per batch
- **`load_best_model_at_end=True`:** Reloads the checkpoint with the highest validation `exact_match`. The trainer saves checkpoints each epoch.

**Evaluation metric:**

Exact match compares predicted and gold SQL after lowercasing and removing space characters:

```python
def normalize_sql(s):
    return s.lower().replace(" ", "").strip()
```

ROUGE-1 is also tracked as a complementary signal for partial matches. Exact match here is a custom string metric. It can treat equivalent SQL strings as different and can collapse differences inside string literals.

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

**Best checkpoint: Epoch 8**, selected by normalized string exact match on the validation subset.

Exact match dipped slightly after epoch 8, while validation loss continued to fall. The evaluation metrics in the table come from the 517-example validation subset used for checkpoint selection. The supplied code does not evaluate the separate holdout.

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

**Setup during the failure:** The base model was loaded directly in BF16 before applying LoRA. The exact cause of the stalled training was not isolated.

**Fix:** Load the base model in the default dtype (fp32), apply LoRA, then manually cast:
- Frozen base weights → `bfloat16` (memory efficiency)
- LoRA adapter weights → `float32` (gradient precision)

The Colab export applies this cast and enables BF16 mixed precision during training. Mixed precision governs computation, while the manual cast sets how the weights are stored.

### 2. LoRA Gradient Flow (Expected Behavior, Mistaken for a Bug)

**Symptom:** At step 0, all `lora_A` weights had zero gradients. Only `lora_B` had non-zero gradients.

**Explanation:** Standard LoRA initialization sets `lora_B = 0`. The adapter contribution is proportional to `lora_B @ lora_A @ x`, so its output starts at zero. The gradient for `lora_A` also starts at zero because it depends on `lora_B`. Once the optimizer updates `lora_B`, `lora_A` can receive a nonzero gradient.

**Verified:** Gradient norms at lora_A across steps: `0.0 → 0.0004 → 0.0012 → 0.0027`. Not a bug.

### 3. `remove_columns` Missing from Dataset Preprocessing

**Symptom:** Training appeared to run but metrics were garbage.

**What I found:** The `remove_columns=dataset['train'].column_names` argument was commented out in the `.map()` call, leaving raw fields such as `question`, `query`, and `db_id` alongside the tokenized columns. The effect of those fields on the earlier metrics was not isolated.

**Fix:** Drop the original columns when tokenizing so the collator receives model inputs and labels:

```python
dataset['train'] = dataset['train'].map(
    preprocess,
    batched=True,
    remove_columns=dataset['train'].column_names,  # critical
)
```

### 4. Out-of-Range Token IDs in Predictions

**Symptom:** Occasional crashes in `compute_metrics` with index errors.

**Possible cause:** FLAN-T5’s output vocabulary can include IDs outside the tokenizer’s vocabulary. The original decoding error was not traced to a specific cause.

**Workaround:** The metric code clips predicted IDs before decoding. This can change the token sequence being scored:

```python
preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
```

---

## Error Analysis

`texttosql.py` includes 12 hand-written questions with example schemas. It prints the generated SQL for qualitative inspection. These examples have no gold SQL labels or automatic scoring.

**Simpler patterns explored:**
- Single-table SELECT with filtering (`WHERE`, `ORDER BY`, `LIMIT`)
- Simple aggregations (`COUNT`, `AVG`, `MAX`)
- Single JOIN with clear column references

**Failure modes to check:**

| Pattern | Example | Notes |
|---|---|---|
| `EXCEPT` / `UNION` / `INTERSECT` | *“Which countries have customers but no orders?”* | Check whether the query correctly excludes countries with orders. |
| 3+ table JOINs | *“What are the names of people who teach math courses?”* | Follow-up example for checking whether the model picks the correct intermediate join table. |
| `HAVING` clause | *“Which departments have more than 5 employees?”* | Check that the query filters groups after aggregation. |
| Repetition loops | `GROUP BY x GROUP BY x GROUP BY x` | Check for duplicated clauses in the generated SQL. |

These cases are useful targets for a larger error analysis. The included probes do not establish how common the failures are or what causes them.

**Note on exact match as a metric:** Different SQL queries can return the same result, so string exact match can miss valid answers. Execution accuracy would test generated queries by running them against the database and comparing the results.

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

- **Larger training data:** Explore whether adding [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) improves coverage of patterns such as set operations, multi-table joins, and `HAVING`
- **Execution-based evaluation:** Replace string exact match with execution accuracy using the official Spider evaluation script
- **Schema linking:** Add a pre-processing step to identify relevant tables/columns before generation, reducing the context complexity and improving JOIN accuracy
- **Larger model:** Evaluate whether `flan-t5-xl` improves performance under the same evaluation protocol

---

## References

- [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Flan-T5: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
