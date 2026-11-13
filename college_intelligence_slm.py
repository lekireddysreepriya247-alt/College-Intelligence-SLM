"""College Data Intelligence SLM - exported notebook source.

This script is an export of the code cells in the accompanying Colab notebook.
Run the notebook in Google Colab for the intended step-by-step workflow.
"""

# # College Data Intelligence – Build a Small Language Model (SLM) with Hugging Face  **Beginner-friendly Google Colab project**  This notebook builds a small, fine-tuned text-to-text model using the uploaded college-program dataset. It also adds semantic retrieval so the assistant can find the relevant record before answering.  **Dataset:** `dataset(1).json` (999 records; 72 fields)  **Model:** `google/flan-t5-small`   **Retrieval model:** `sentence-transformers/all-MiniLM-L6-v2`  > Important: the JSON file is structured tabular data, not a ready-made chat dataset. This notebook creates question-answer training examples from available fields. It splits by institution identifier to reduce institution-level leakage. A separate test set is kept for validation. The language model is a small general pretrained model that we fine-tune; it is not trained from random weights.  **Recommended:** Google Colab → Runtime → Change runtime type → T4 GPU (if available).

# ## Step 1 – Set up the environment  Run this cell first. It installs the libraries used for data handling, fine-tuning, embeddings, and evaluation.

# ---- Notebook code cell 3 ----
!pip -q install -U transformers datasets accelerate sentence-transformers scikit-learn pandas numpy huggingface_hub

# ## Step 2 – Upload your JSON dataset  When the file picker appears, select the `dataset(1).json` file you uploaded in ChatGPT.  If you have already uploaded it to the Colab session, you can skip the upload cell and set `DATA_PATH` to its location.

# ---- Notebook code cell 5 ----
from google.colab import files
uploaded = files.upload()  # Select dataset(1).json
DATA_PATH = next(iter(uploaded.keys()))
print("Using:", DATA_PATH)

# ## Step 3 – Load and inspect the dataset  This checks the file, number of records, columns, and missing values. The JSON is expected to be a list of objects.

# ---- Notebook code cell 7 ----
import json
import pandas as pd
import numpy as np

with open(DATA_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

if not isinstance(records, list) or len(records) == 0:
    raise ValueError("Expected a non-empty JSON list of records.")

df = pd.DataFrame(records)
print("Rows:", len(df))
print("Columns:", len(df.columns))
display(df.head(3))
print("\\nMissing-value percentage (top 15 columns):")
display((df.isna().mean().sort_values(ascending=False).head(15) * 100).round(1))

# ## Step 4 – Clean values and prepare useful text fields  Missing values are kept as missing rather than being replaced with invented facts. The `program_id.2` field is used as the institution grouping key when available; otherwise, institution name is used.

# ---- Notebook code cell 9 ----
def is_missing(x):
    try:
        return pd.isna(x)
    except Exception:
        return x is None

def clean_value(x):
    if is_missing(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if not np.isfinite(x):
            return None
        return float(x)
    return str(x).strip()

def money(x):
    if is_missing(x):
        return None
    return f"${float(x):,.0f}"

# Build stable institution groups for leakage-resistant splitting.
def get_group(row):
    inst_id = clean_value(row.get("program_id.2"))
    if inst_id is not None:
        return str(inst_id)
    name = clean_value(row.get("program_id.3"))
    return str(name) if name else str(row.get("program_id", "unknown"))

df["group_id"] = df.apply(get_group, axis=1)
print("Unique institution groups:", df["group_id"].nunique())
print("Example institution:", df.iloc[0].get("program_id.3"))
print("Example program:", df.iloc[0].get("cip_title"))

# ## Step 5 – Create supervised question-answer examples  A language model learns from input/target examples. We generate factual questions from fields that actually exist in each row.  Examples include: - Which college offers this program? - What credential does this program award? - What is the in-state tuition? - What are the reported four-year median earnings? - What is the median student debt? - Which occupation is most strongly linked to the program?  Rows with missing values for a target field are skipped for that particular question.

# ---- Notebook code cell 11 ----
def make_context(row, target_label, target_value):
    parts = []
    fields = [
        ("Institution", "program_id.3"),
        ("City", "program_id.6"),
        ("State", "program_id.7"),
        ("Institution type", "program_id.4"),
        ("Program", "cip_title"),
        ("Credential", "credential_name"),
    ]
    for label, col in fields:
        value = clean_value(row.get(col))
        if value is not None:
            parts.append(f"{label}: {value}")
    parts.append(f"{target_label}: {target_value}")
    return " | ".join(parts)

def build_qa_examples(frame):
    examples = []
    for row_idx, row in frame.iterrows():
        college = clean_value(row.get("program_id.3")) or "this institution"
        program = clean_value(row.get("cip_title")) or "this program"
        credential = clean_value(row.get("credential_name"))

        targets = [
            ("institution", "Which institution offers the program?", lambda r: clean_value(r.get("program_id.3"))),
            ("program", "What is the name of the study program?", lambda r: clean_value(r.get("cip_title"))),
            ("credential", "What credential does this program award?", lambda r: clean_value(r.get("credential_name"))),
            ("in_state_tuition", "What is the in-state tuition for this program?", lambda r: money(r.get("institution_tuition_in_state_usd"))),
            ("out_state_tuition", "What is the out-of-state tuition for this program?", lambda r: money(r.get("institution_tuition_out_state_usd"))),
            ("four_year_earnings", "What are the median earnings four years after completion?", lambda r: money(r.get("median_earnings_4yr_usd"))),
            ("median_debt", "What is the median student debt for this program?", lambda r: money(r.get("median_debt_usd"))),
            ("occupation", "What is the largest linked occupation?", lambda r: clean_value(r.get("largest_linked_occupation"))),
            ("occupation_wage", "What is the linked occupation's median annual wage?", lambda r: money(r.get("occupation_median_wage_2024_usd"))),
            ("earnings_trajectory", "What is the earnings trajectory category?", lambda r: clean_value(r.get("earnings_trajectory_category"))),
            ("ai_tools", "Which AI tools are listed for linked occupations?", lambda r: clean_value(r.get("ai_tool_examples"))),
        ]

        for key, question, getter in targets:
            answer = getter(row)
            if answer is None or str(answer).strip() == "":
                continue
            label_map = {
                "institution": "Institution", "program": "Program",
                "credential": "Credential", "in_state_tuition": "In-state tuition",
                "out_state_tuition": "Out-of-state tuition",
                "four_year_earnings": "Median earnings four years after completion",
                "median_debt": "Median student debt",
                "occupation": "Largest linked occupation",
                "occupation_wage": "Linked occupation median annual wage",
                "earnings_trajectory": "Earnings trajectory category",
                "ai_tools": "AI tools listed"
            }
            context = make_context(row, label_map[key], answer)
            prompt = (
                "Answer the question using only the information in the context. "
                "Do not guess. If the information is not present, say it is not reported. "
                f"Context: {context} Question: {question}"
            )
            examples.append({
                "row_index": int(row_idx),
                "group_id": str(row["group_id"]),
                "question": question,
                "input_text": prompt,
                "target_text": str(answer)
            })
    return pd.DataFrame(examples)

qa_df = build_qa_examples(df)
print("Generated QA examples:", len(qa_df))
display(qa_df[["question", "input_text", "target_text"]].head(5))

# ## Step 6 – Split into training, validation, and testing sets  We split by institution group, not by individual question. This helps prevent the same institution appearing in both training and test data.  Target split: approximately 80% training, 10% validation, and 10% testing. Exact percentages can vary because whole institutions are kept together.

# ---- Notebook code cell 13 ----
from sklearn.model_selection import GroupShuffleSplit

row_meta = df[["group_id"]].copy()
row_meta["row_index"] = row_meta.index

# First hold out 10% of institution groups for final testing.
split1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
trainval_idx, test_idx = next(split1.split(row_meta, groups=row_meta["group_id"]))

trainval_meta = row_meta.iloc[trainval_idx].copy()
# Hold out 1/9 of the remaining groups, approximately 10% of all data, for validation.
split2 = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
train_rel, val_rel = next(split2.split(trainval_meta, groups=trainval_meta["group_id"]))

train_rows = set(trainval_meta.iloc[train_rel]["row_index"].tolist())
val_rows = set(trainval_meta.iloc[val_rel]["row_index"].tolist())
test_rows = set(row_meta.iloc[test_idx]["row_index"].tolist())

train_df = qa_df[qa_df["row_index"].isin(train_rows)].reset_index(drop=True)
val_df = qa_df[qa_df["row_index"].isin(val_rows)].reset_index(drop=True)
test_df = qa_df[qa_df["row_index"].isin(test_rows)].reset_index(drop=True)

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))
print("Testing examples:", len(test_df))
print("Institution overlap train/test:", len(set(df.loc[list(train_rows), "group_id"]) & set(df.loc[list(test_rows), "group_id"])))
assert len(train_df) and len(val_df) and len(test_df), "A split is empty. Check the number of institution groups."

# ## Step 7 – Load a small Hugging Face model  We use `google/flan-t5-small`, a compact instruction-following text-to-text model. It is suitable for a first Colab experiment and much lighter than large chat models.  This is the base model before fine-tuning.

# ---- Notebook code cell 15 ----
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Base model loaded:", MODEL_NAME)

# ## Step 8 – Tokenize the data for training  Tokenization converts text into numerical tokens the model can process. We keep the input length bounded to fit Colab memory.

# ---- Notebook code cell 17 ----
from datasets import Dataset

MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 64

def tokenize_batch(batch):
    inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

train_ds = Dataset.from_pandas(train_df[["input_text", "target_text"]], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[["input_text", "target_text"]], preserve_index=False)
test_ds = Dataset.from_pandas(test_df[["input_text", "target_text"]], preserve_index=False)

train_tok = train_ds.map(tokenize_batch, batched=True, remove_columns=train_ds.column_names)
val_tok = val_ds.map(tokenize_batch, batched=True, remove_columns=val_ds.column_names)
test_tok = test_ds.map(tokenize_batch, batched=True, remove_columns=test_ds.column_names)

print(train_tok)

# ## Step 9 – Fine-tune (train) the SLM  This is the actual model-training step. The model learns to answer questions using the supplied record context.  If Colab runs out of memory: 1. Change `BATCH_SIZE` from 8 to 4 or 2. 2. Reduce `MAX_INPUT_LENGTH` to 256. 3. Make sure a GPU runtime is selected.

# ---- Notebook code cell 19 ----
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
BATCH_SIZE = 8 if torch.cuda.is_available() else 2

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/college_slm_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    num_train_epochs=3,
    predict_with_generate=True,
    generation_max_length=64,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    processing_class=tokenizer
)

train_result = trainer.train()
print("Training completed.")

# ## Step 10 – Validate the fine-tuned model on unseen test examples  We evaluate on the held-out test set that was not used for training or model selection. The notebook reports: - Test loss - Exact-match rate (strict: the generated answer must match after basic normalization) - Token F1 (partial word overlap)  Because natural-language generation can paraphrase an answer, exact match is strict. Always inspect example predictions as well.

# ---- Notebook code cell 21 ----
import re
from collections import Counter

test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")
print("Held-out test loss:", test_metrics.get("test_loss"))

def normalize_text(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9$.,% ]", "", s)
    s = re.sub(r"\\s+", " ", s)
    return s

def token_f1(pred, gold):
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)

# Generate predictions for up to 200 held-out test examples to keep runtime reasonable.
eval_sample = test_df.head(min(200, len(test_df))).copy()
predictions = []
for _, ex in eval_sample.iterrows():
    inputs = tokenizer(ex["input_text"], return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_LENGTH).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    predictions.append(tokenizer.decode(out[0], skip_special_tokens=True))

eval_sample["prediction"] = predictions
eval_sample["exact_match"] = [
    normalize_text(p) == normalize_text(g)
    for p, g in zip(eval_sample["prediction"], eval_sample["target_text"])
]
eval_sample["token_f1"] = [
    token_f1(p, g) for p, g in zip(eval_sample["prediction"], eval_sample["target_text"])
]

print("Test examples scored:", len(eval_sample))
print("Exact match:", round(eval_sample["exact_match"].mean(), 4))
print("Average token F1:", round(eval_sample["token_f1"].mean(), 4))
display(eval_sample[["question", "target_text", "prediction", "exact_match", "token_f1"]].head(15))

# ## Step 11 – Save your trained model  This saves the fine-tuned model and tokenizer to your Colab files. You can download the folder as a ZIP.

# ---- Notebook code cell 23 ----
SAVE_DIR = "/content/college-data-slm"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

import shutil
zip_path = shutil.make_archive("/content/college-data-slm", "zip", SAVE_DIR)
print("Saved model folder:", SAVE_DIR)
print("ZIP file:", zip_path)

# ## Step 12 – Build semantic search over the dataset  The retriever finds the most relevant program record for a user's question. The fine-tuned SLM then answers using that record as context.  For the deployed assistant, the full dataset is indexed because it is the knowledge base. The held-out test evaluation above remains separate and is not used to train the model weights.

# ---- Notebook code cell 25 ----
from sentence_transformers import SentenceTransformer, util

def row_to_search_text(row):
    fields = [
        ("Institution", "program_id.3"),
        ("City", "program_id.6"),
        ("State", "program_id.7"),
        ("Institution type", "program_id.4"),
        ("Program", "cip_title"),
        ("Credential", "credential_name"),
        ("In-state tuition", "institution_tuition_in_state_usd"),
        ("Out-of-state tuition", "institution_tuition_out_state_usd"),
        ("Four-year median earnings", "median_earnings_4yr_usd"),
        ("Median debt", "median_debt_usd"),
        ("Largest linked occupation", "largest_linked_occupation"),
        ("Occupation median wage", "occupation_median_wage_2024_usd"),
        ("AI tools", "ai_tool_examples"),
    ]
    parts = []
    for label, col in fields:
        value = clean_value(row.get(col))
        if value is not None:
            if "tuition" in label.lower() or "earnings" in label.lower() or "debt" in label.lower() or "wage" in label.lower():
                value = money(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
            parts.append(f"{label}: {value}")
    return " | ".join(parts)

search_texts = [row_to_search_text(row) for _, row in df.iterrows()]
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
corpus_embeddings = embedder.encode(search_texts, convert_to_tensor=True, show_progress_bar=True)
print("Indexed records:", len(search_texts))

# ## Step 13 – Create the chatbot function  Ask a question about an institution, program, tuition, earnings, debt, or linked occupation. The retriever selects the closest record and the fine-tuned SLM generates a response.  The answer is only as reliable as the retrieved row and the source data. If the field is missing or suppressed, the assistant should say it is not reported.

# ---- Notebook code cell 27 ----
def ask_college_slm(question, top_k=3):
    q_embedding = embedder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_embedding, corpus_embeddings, top_k=top_k)[0]

    if not hits:
        return "I could not find a matching record in the dataset."

    best = hits[0]
    row_index = best["corpus_id"]
    row = df.iloc[row_index]

    # Guard against inventing values when a requested field is missing/suppressed.
    q = question.lower()
    checks = [
        (["in-state tuition", "in state tuition"], "institution_tuition_in_state_usd", "In-state tuition"),
        (["out-of-state tuition", "out of state tuition"], "institution_tuition_out_state_usd", "Out-of-state tuition"),
        (["tuition", "fees"], "institution_tuition_in_state_usd", "Tuition"),
        (["four-year earnings", "4-year earnings", "four year earnings", "4 year earnings"], "median_earnings_4yr_usd", "Four-year median earnings"),
        (["earnings"], "median_earnings_4yr_usd", "Four-year median earnings"),
        (["median debt", "student debt", "debt"], "median_debt_usd", "Median student debt"),
        (["occupation wage", "occupation salary", "median wage", "annual wage"], "occupation_median_wage_2024_usd", "Occupation median annual wage"),
    ]
    for terms, field, label in checks:
        if any(term in q for term in terms):
            value = clean_value(row.get(field))
            if value is None:
                status_field = "earnings_4yr_status" if field == "median_earnings_4yr_usd" else (
                    "debt_status" if field == "median_debt_usd" else None
                )
                status = clean_value(row.get(status_field)) if status_field else None
                suffix = f" Status: {status}." if status else ""
                return {
                    "answer": f"{label} is not reported in the matched dataset record.{suffix}",
                    "matched_institution": clean_value(row.get("program_id.3")),
                    "matched_program": clean_value(row.get("cip_title")),
                    "similarity_score": round(float(best["score"]), 4),
                    "other_matches": []
                }

    context = row_to_search_text(row)
    prompt = (
        "Answer the question using only the context. Do not invent facts. "
        "If the requested value is missing, suppressed, or not present, say it is not reported. "
        f"Context: {context} Question: {question}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_LENGTH).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, num_beams=4)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "matched_institution": clean_value(row.get("program_id.3")),
        "matched_program": clean_value(row.get("cip_title")),
        "similarity_score": round(float(best["score"]), 4),
        "other_matches": [
            {
                "institution": clean_value(df.iloc[h["corpus_id"]].get("program_id.3")),
                "program": clean_value(df.iloc[h["corpus_id"]].get("cip_title")),
                "score": round(float(h["score"]), 4)
            } for h in hits[1:]
        ]
    }

# Try these examples:
print(ask_college_slm("What is the four-year median earnings for a mathematics bachelor's program?"))

# ## Step 14 – Try your own questions  Run the cell, type a question, and press Enter. Try questions about a named college and program to make retrieval more precise.

# ---- Notebook code cell 29 ----
while True:
    question = input("Ask about college/program data (type exit to stop): ").strip()
    if question.lower() in {"exit", "quit", "stop"}:
        print("Chat ended.")
        break
    if question:
        result = ask_college_slm(question)
        print(result)
        print("-" * 80)

# ## Step 15 – Publish the model to Hugging Face Hub (optional)  You need a free Hugging Face account and an access token with write permission.  1. Create an account at https://huggingface.co 2. Open Settings → Access Tokens → create a token with write access. 3. Run the cell below and paste your token when prompted. 4. Choose a unique repository name.  Do not share your token or put it in a public notebook.

# ---- Notebook code cell 31 ----
# OPTIONAL – uncomment and run only when you are ready to upload.
# The repository is created as PRIVATE so your fine-tuned model is not public by default.
#
# from huggingface_hub import notebook_login, create_repo
# notebook_login()  # Enter your Hugging Face write token in the secure prompt.
#
# REPO_ID = "YOUR_HUGGINGFACE_USERNAME/college-data-flan-t5-small"
# create_repo(REPO_ID, private=True, exist_ok=True)
# trainer.model.push_to_hub(REPO_ID)
# tokenizer.push_to_hub(REPO_ID)
# print("Uploaded private model:", REPO_ID)

# ## Step 16 – What to report in your project  Record the actual values printed by your run; do not invent accuracy numbers.  Include: 1. Dataset size and fields (999 records, 72 columns). 2. Preprocessing and missing-value handling. 3. Institution-grouped training/validation/testing split. 4. Base model: `google/flan-t5-small`. 5. Fine-tuning settings: epochs, batch size, learning rate. 6. Test loss, exact-match rate, and token F1. 7. Example questions, expected answers, and model predictions. 8. Limitations: synthetic QA pairs, small dataset, possible retrieval mismatch, and missing/suppressed values.  **Important limitation:** this is a student prototype, not a verified college-admissions or financial-advice system. The dataset contains US-dollar values and may contain suppressed or missing outcome fields. Do not treat absent values as zero.
