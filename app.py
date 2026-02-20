# Gradio app

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# --- Model Loading ---

BASE_MODEL = "google/flan-t5-large"
ADAPTER_MODEL = "artmiss/flan-t5-large-spider-text2sql"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.eval()
print("Model loaded.")

# --- Inference ---

def predict(question: str, schema: str) -> str:
    if not question.strip():
        return "⚠️ Please enter a question."
    if not schema.strip():
        return "⚠️ Please add at least one table to the schema."

    input_text = f"Translate English to SQL: {question} | Schemas: {schema}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- Schema Builder Logic ---

def build_schema(tables_state):
    """Convert the tables state dict into schema string: table(col1, col2) | table2(col3)"""
    parts = []
    for table_name, columns in tables_state.items():
        if table_name.strip():
            cols = [c.strip() for c in columns if c.strip()]
            if cols:
                parts.append(f"{table_name.strip()}({', '.join(cols)})")
    return " | ".join(parts)


def add_table(tables_state, new_table_name):
    name = new_table_name.strip()
    if not name:
        return tables_state, gr.update(), format_schema_display(tables_state), "⚠️ Table name cannot be empty."
    if name in tables_state:
        return tables_state, gr.update(), format_schema_display(tables_state), f"⚠️ Table '{name}' already exists."
    tables_state[name] = []
    return tables_state, gr.update(value=""), format_schema_display(tables_state), f"✅ Table '{name}' added."


def add_column(tables_state, selected_table, new_col_name):
    col = new_col_name.strip()
    if not selected_table:
        return tables_state, gr.update(), format_schema_display(tables_state), "⚠️ Select a table first."
    if not col:
        return tables_state, gr.update(), format_schema_display(tables_state), "⚠️ Column name cannot be empty."
    if col in tables_state.get(selected_table, []):
        return tables_state, gr.update(), format_schema_display(tables_state), f"⚠️ Column '{col}' already exists in '{selected_table}'."
    tables_state[selected_table].append(col)
    return tables_state, gr.update(value=""), format_schema_display(tables_state), f"✅ Column '{col}' added to '{selected_table}'."


def remove_table(tables_state, selected_table):
    if not selected_table or selected_table not in tables_state:
        return tables_state, gr.update(choices=list(tables_state.keys()), value=None), format_schema_display(tables_state), "⚠️ Select a table to remove."
    del tables_state[selected_table]
    choices = list(tables_state.keys())
    return tables_state, gr.update(choices=choices, value=choices[0] if choices else None), format_schema_display(tables_state), f"🗑️ Table '{selected_table}' removed."


def update_table_dropdown(tables_state):
    return gr.update(choices=list(tables_state.keys()), value=list(tables_state.keys())[0] if tables_state else None)


def format_schema_display(tables_state):
    if not tables_state:
        return "_No tables added yet._"
    lines = []
    for table, cols in tables_state.items():
        col_str = ", ".join(cols) if cols else "_no columns_"
        lines.append(f"**{table}** ( {col_str} )")
    return "\n\n".join(lines)


def run_prediction(question, tables_state):
    schema = build_schema(tables_state)
    sql = predict(question, schema)
    return sql


def load_example(example, tables_state):
    question = example[0]
    schema_str = example[1]
    # Parse schema string back into tables_state
    new_state = {}
    for part in schema_str.split(" | "):
        if "(" in part and part.endswith(")"):
            table_name = part[:part.index("(")].strip()
            cols_str = part[part.index("(")+1:-1]
            cols = [c.strip() for c in cols_str.split(",") if c.strip()]
            new_state[table_name] = cols
    return question, new_state, gr.update(choices=list(new_state.keys()), value=list(new_state.keys())[0] if new_state else None), format_schema_display(new_state)


# --- Examples ---

EXAMPLES = [
    [
        "How many players are from each country?",
        "players(player_id, first_name, last_name, country_code, birth_date)",
    ],
    [
        "Who are the top 3 highest paid employees?",
        "employees(employee_id, name, age, salary, department_id)",
    ],
    [
        "What are the names of customers who placed an order?",
        "customers(customer_id, name, email, country) | orders(order_id, customer_id, total, date)",
    ],
    [
        "What is the average salary of employees in each department?",
        "employees(employee_id, name, salary, department_id) | departments(department_id, name, location)",
    ],
    [
        "Which products cost more than 100?",
        "products(product_id, name, price, category, stock)",
    ],
]

# --- UI ---

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #13161d;
    --surface2: #1a1e28;
    --border: #252a38;
    --accent: #4fffb0;
    --accent2: #4d9eff;
    --text: #e2e8f0;
    --muted: #64748b;
    --sql-bg: #0a0c10;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.app-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--accent);
    margin: 0;
    line-height: 1.1;
}

.app-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.panel-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
}

/* Inputs */
input[type="text"], textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79, 255, 176, 0.1) !important;
    outline: none !important;
}

/* Buttons */
button.primary-btn {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    letter-spacing: 0.02em !important;
}

button.primary-btn:hover {
    opacity: 0.85 !important;
}

button.secondary-btn {
    background: var(--surface2) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    cursor: pointer !important;
    transition: border-color 0.15s !important;
}

button.secondary-btn:hover {
    border-color: var(--accent2) !important;
}

button.danger-btn {
    background: transparent !important;
    color: #f87171 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    border: 1px solid #3d1f1f !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    cursor: pointer !important;
}

/* SQL output */
.sql-output {
    background: var(--sql-bg) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 8px !important;
    padding: 1.1rem 1.25rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    color: var(--accent) !important;
    min-height: 60px;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Schema display */
.schema-display {
    background: var(--sql-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--accent2) !important;
    min-height: 80px;
}

/* Status messages */
.status-msg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    min-height: 1.2rem;
    padding: 0.25rem 0;
}

/* Examples */
.example-btn {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 0.75rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    text-align: left !important;
}

.example-btn:hover {
    border-color: var(--accent2) !important;
    color: var(--text) !important;
}

/* Dropdown */
select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Labels */
label span {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Syne', sans-serif !important;
}

/* Markdown */
.schema-display p, .schema-display strong {
    color: var(--accent2) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide gradio footer */
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Text-to-SQL") as demo:

    tables_state = gr.State({})

    # Header
    gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">Text → SQL</h1>
            <p class="app-subtitle">flan-t5-large · LoRA · Spider benchmark</p>
        </div>
    """)

    with gr.Row():

        # Left column — Schema Builder
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-title">Schema Builder</div>')

            with gr.Group():
                new_table_input = gr.Textbox(
                    placeholder="e.g. players",
                    label="Table name",
                    lines=1,
                )
                add_table_btn = gr.Button("+ Add Table", elem_classes=["secondary-btn"])

            with gr.Group():
                table_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select table",
                    interactive=True,
                )
                new_col_input = gr.Textbox(
                    placeholder="e.g. player_id",
                    label="Column name",
                    lines=1,
                )
                with gr.Row():
                    add_col_btn = gr.Button("+ Add Column", elem_classes=["secondary-btn"])
                    remove_table_btn = gr.Button("Remove Table", elem_classes=["danger-btn"])

            gr.HTML('<div class="panel-title" style="margin-top:1rem">Current Schema</div>')
            schema_display = gr.Markdown(
                value="_No tables added yet._",
                elem_classes=["schema-display"],
            )
            status_msg = gr.Markdown(value="", elem_classes=["status-msg"])

        # Right column — Question + Output
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-title">Question</div>')
            question_input = gr.Textbox(
                placeholder="e.g. How many players are from each country?",
                label="Natural language question",
                lines=3,
            )
            generate_btn = gr.Button("Generate SQL →", elem_classes=["primary-btn"])

            gr.HTML('<div class="panel-title" style="margin-top:1.5rem">Generated SQL</div>')
            sql_output = gr.Code(
                label="",
                language="sql",
                lines=5,
                interactive=False,
            )

    # Examples
    gr.HTML('<div class="panel-title" style="margin-top:1.5rem">Examples</div>')
    with gr.Row():
        for ex in EXAMPLES:
            ex_btn = gr.Button(ex[0][:45] + ("…" if len(ex[0]) > 45 else ""), elem_classes=["example-btn"])
            ex_btn.click(
                fn=lambda e=ex: load_example(e, {}),
                inputs=[],
                outputs=[question_input, tables_state, table_dropdown, schema_display],
            )

    # --- Event wiring ---

    add_table_btn.click(
        fn=add_table,
        inputs=[tables_state, new_table_input],
        outputs=[tables_state, new_table_input, schema_display, status_msg],
    ).then(
        fn=update_table_dropdown,
        inputs=[tables_state],
        outputs=[table_dropdown],
    )

    add_col_btn.click(
        fn=add_column,
        inputs=[tables_state, table_dropdown, new_col_input],
        outputs=[tables_state, new_col_input, schema_display, status_msg],
    )

    remove_table_btn.click(
        fn=remove_table,
        inputs=[tables_state, table_dropdown],
        outputs=[tables_state, table_dropdown, schema_display, status_msg],
    )

    generate_btn.click(
        fn=run_prediction,
        inputs=[question_input, tables_state],
        outputs=[sql_output],
    )

    new_table_input.submit(
        fn=add_table,
        inputs=[tables_state, new_table_input],
        outputs=[tables_state, new_table_input, schema_display, status_msg],
    ).then(
        fn=update_table_dropdown,
        inputs=[tables_state],
        outputs=[table_dropdown],
    )

    new_col_input.submit(
        fn=add_column,
        inputs=[tables_state, table_dropdown, new_col_input],
        outputs=[tables_state, new_col_input, schema_display, status_msg],
    )

    question_input.submit(
        fn=run_prediction,
        inputs=[question_input, tables_state],
        outputs=[sql_output],
    )


if __name__ == "__main__":
    demo.launch()