# this version matches the color and format of huggingface sapces

import gradio as gr
import torch
import re
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

# --- SQL Syntax Highlighter ---

SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL",
    "ON", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "DISTINCT",
    "COUNT", "SUM", "AVG", "MAX", "MIN", "AS", "AND", "OR", "NOT", "IN",
    "EXISTS", "BETWEEN", "LIKE", "IS", "NULL", "INSERT", "INTO", "VALUES",
    "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "INDEX",
    "UNION", "INTERSECT", "EXCEPT", "CASE", "WHEN", "THEN", "ELSE", "END",
    "ASC", "DESC", "WITH", "RECURSIVE",
]

def highlight_sql(sql: str) -> str:
    if not sql or sql.startswith("⚠️"):
        color = "#f87171" if sql.startswith("⚠️") else "#64748b"
        return f'<div style="background:#0a0c10;border:1px solid #252a38;border-left:3px solid #252a38;border-radius:8px;padding:1rem 1.25rem;min-height:56px;font-family:monospace;font-size:0.9rem;"><span style="color:{color};">{sql}</span></div>'

    sql_escaped = sql.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tokens = re.split(r'(\s+)', sql_escaped)
    result = []
    for token in tokens:
        stripped = token.strip("(),;.*")
        upper = stripped.upper()
        if upper in SQL_KEYWORDS:
            result.append(f'<span style="color:#4fffb0;font-weight:600;">{token}</span>')
        elif re.match(r"^'[^']*'$", stripped) or re.match(r'^"[^"]*"$', stripped):
            result.append(f'<span style="color:#fbbf24;">{token}</span>')
        elif re.match(r'^\d+(\.\d+)?$', stripped):
            result.append(f'<span style="color:#f472b6;">{token}</span>')
        elif re.match(r'^T\d+$', stripped):
            result.append(f'<span style="color:#a78bfa;">{token}</span>')
        else:
            result.append(f'<span style="color:#e2e8f0;">{token}</span>')

    inner = "".join(result)
    return (
        '<div style="'
        'background:#0a0c10;'
        'border:1px solid #252a38;'
        'border-left:3px solid #4fffb0;'
        'border-radius:8px;'
        'padding:1rem 1.25rem;'
        'font-family:monospace;'
        'font-size:0.92rem;'
        'line-height:1.7;'
        'min-height:56px;'
        'white-space:pre-wrap;'
        'word-break:break-word;'
        f'">{inner}</div>'
    )


# --- Inference ---

def predict(question: str, schema: str) -> str:
    if not question.strip():
        return "⚠️ Please enter a question."
    if not schema.strip():
        return "⚠️ Please add at least one table to the schema."

    input_text = f"Translate English to SQL: {question} | Schemas: {schema}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- Schema Builder Logic ---

def build_schema(tables_state):
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
    return highlight_sql(sql)


def load_example(example, tables_state):
    question = example[0]
    schema_str = example[1]
    new_state = {}
    for part in schema_str.split(" | "):
        if "(" in part and part.endswith(")"):
            table_name = part[:part.index("(")].strip()
            cols_str = part[part.index("(")+1:-1]
            cols = [c.strip() for c in cols_str.split(",") if c.strip()]
            new_state[table_name] = cols
    return (
        question,
        new_state,
        gr.update(choices=list(new_state.keys()), value=list(new_state.keys())[0] if new_state else None),
        format_schema_display(new_state),
    )


# --- Examples ---

EXAMPLES = [
    ["How many players are from each country?",
     "players(player_id, first_name, last_name, country_code, birth_date)"],
    ["Who are the top 3 highest paid employees?",
     "employees(employee_id, name, age, salary, department_id)"],
    ["What are the names of customers who placed an order?",
     "customers(customer_id, name, email, country) | orders(order_id, customer_id, total, date)"],
    ["What is the average salary of employees in each department?",
     "employees(employee_id, name, salary, department_id) | departments(department_id, name, location)"],
    ["Which products cost more than 100?",
     "products(product_id, name, price, category, stock)"],
]

# --- CSS ---

CSS = """
:root {
    --bg:      #0d0f14;
    --surface: #13161d;
    --surface2:#1a1e28;
    --border:  #252a38;
    --accent:  #4fffb0;
    --accent2: #4d9eff;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    --sans: 'DM Sans', system-ui, sans-serif;
}

body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }

.gradio-container, .gradio-container > div,
.block, .wrap, .gap, .form, .tabs, .tabitem, .panel, .prose {
    background: transparent !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

input, textarea, select, input[type="text"], input[type="search"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.875rem !important;
    caret-color: var(--accent) !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,255,176,0.12) !important;
    outline: none !important;
}

label > span, .label-wrap span {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-family: var(--sans) !important;
}

button { font-family: var(--sans) !important; }

button.primary-btn {
    background: var(--accent) !important;
    color: #0a0c10 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
    width: 100% !important;
}
button.primary-btn:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

button.secondary-btn {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: border-color 0.15s !important;
    width: 100% !important;
}
button.secondary-btn:hover { border-color: var(--accent2) !important; color: var(--accent2) !important; }

button.danger-btn {
    background: transparent !important;
    color: #f87171 !important;
    border: 1px solid #3d1f1f !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: background 0.15s !important;
    width: 100% !important;
}
button.danger-btn:hover { background: rgba(248,113,113,0.08) !important; }

button.example-btn {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    padding: 0.45rem 0.8rem !important;
    text-align: left !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
button.example-btn:hover { border-color: var(--accent2) !important; color: var(--text) !important; }

.wrap-inner, .multiselect, ul.options {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
ul.options li { background: var(--surface2) !important; color: var(--text) !important; }
ul.options li:hover, ul.options li.selected { background: var(--surface) !important; color: var(--accent) !important; }

.schema-display {
    background: #0a0c10 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
    min-height: 80px;
}
.schema-display p { color: var(--accent2) !important; font-family: var(--mono) !important; font-size: 0.82rem !important; margin: 0.2rem 0 !important; }
.schema-display strong { color: var(--accent2) !important; }

.status-msg p { font-family: var(--mono) !important; font-size: 0.78rem !important; color: var(--muted) !important; margin: 0 !important; }

.panel-title {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
    font-family: var(--sans);
}

.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.75rem;
}
.app-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: var(--accent);
    margin: 0;
    font-family: var(--sans);
    line-height: 1;
}
.app-subtitle {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.5rem;
    font-family: var(--mono);
}

footer { display: none !important; }
"""

# --- App ---

with gr.Blocks(
    css=CSS,
    title="Text-to-SQL",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0d0f14",
        body_background_fill_dark="#0d0f14",
        block_background_fill="#13161d",
        block_background_fill_dark="#13161d",
        block_border_color="#252a38",
        block_border_color_dark="#252a38",
        input_background_fill="#1a1e28",
        input_background_fill_dark="#1a1e28",
        input_border_color="#252a38",
        input_border_color_dark="#252a38",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        button_secondary_background_fill="#1a1e28",
        button_secondary_background_fill_dark="#1a1e28",
        button_secondary_border_color="#252a38",
        button_secondary_border_color_dark="#252a38",
        button_secondary_text_color="#e2e8f0",
        button_secondary_text_color_dark="#e2e8f0",
    )
) as demo:

    tables_state = gr.State({})

    gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">Text &rarr; SQL</h1>
            <p class="app-subtitle">flan-t5-large &middot; LoRA &middot; Spider benchmark &middot</p>
        </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):
            gr.HTML('<div class="panel-title">Schema Builder</div>')
            with gr.Group():
                new_table_input = gr.Textbox(placeholder="e.g. players", label="Table name", lines=1)
                add_table_btn = gr.Button("+ Add Table", elem_classes=["secondary-btn"])
            with gr.Group():
                table_dropdown = gr.Dropdown(choices=[], label="Select table", interactive=True)
                new_col_input = gr.Textbox(placeholder="e.g. player_id", label="Column name", lines=1)
                with gr.Row():
                    add_col_btn = gr.Button("+ Add Column", elem_classes=["secondary-btn"])
                    remove_table_btn = gr.Button("Remove Table", elem_classes=["danger-btn"])
            gr.HTML('<div class="panel-title" style="margin-top:1.2rem">Current Schema</div>')
            schema_display = gr.Markdown(value="_No tables added yet._", elem_classes=["schema-display"])
            status_msg = gr.Markdown(value="", elem_classes=["status-msg"])

        with gr.Column(scale=1):
            gr.HTML('<div class="panel-title">Question</div>')
            question_input = gr.Textbox(
                placeholder="e.g. How many players are from each country?",
                label="Natural language question",
                lines=3,
            )
            generate_btn = gr.Button("Generate SQL →", elem_classes=["primary-btn"])
            gr.HTML('<div class="panel-title" style="margin-top:1.5rem">Generated SQL</div>')
            sql_output = gr.HTML(
                value='<div style="background:#0a0c10;border:1px solid #252a38;border-left:3px solid #252a38;border-radius:8px;padding:1rem 1.25rem;min-height:56px;font-family:monospace;color:#64748b;font-size:0.88rem;">Output will appear here...</div>'
            )

    gr.HTML('<div class="panel-title" style="margin-top:1.5rem">Examples</div>')
    with gr.Row():
        for ex in EXAMPLES:
            ex_btn = gr.Button(
                ex[0][:48] + ("…" if len(ex[0]) > 48 else ""),
                elem_classes=["example-btn"]
            )
            ex_btn.click(
                fn=lambda e=ex: load_example(e, {}),
                inputs=[],
                outputs=[question_input, tables_state, table_dropdown, schema_display],
            )

    add_table_btn.click(
        fn=add_table,
        inputs=[tables_state, new_table_input],
        outputs=[tables_state, new_table_input, schema_display, status_msg],
    ).then(fn=update_table_dropdown, inputs=[tables_state], outputs=[table_dropdown])

    new_table_input.submit(
        fn=add_table,
        inputs=[tables_state, new_table_input],
        outputs=[tables_state, new_table_input, schema_display, status_msg],
    ).then(fn=update_table_dropdown, inputs=[tables_state], outputs=[table_dropdown])

    add_col_btn.click(
        fn=add_column,
        inputs=[tables_state, table_dropdown, new_col_input],
        outputs=[tables_state, new_col_input, schema_display, status_msg],
    )

    new_col_input.submit(
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

    question_input.submit(
        fn=run_prediction,
        inputs=[question_input, tables_state],
        outputs=[sql_output],
    )


if __name__ == "__main__":
    demo.launch()