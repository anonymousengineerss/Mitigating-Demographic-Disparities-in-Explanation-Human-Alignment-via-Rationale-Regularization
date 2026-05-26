import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = "/scratch/gilbreth/talusb01/EMNLP"
DATA_DIR = BASE_DIR
OUT_DIR = os.path.join(BASE_DIR, "bert_output")
os.makedirs(OUT_DIR, exist_ok=True)

# HF offline cache
os.environ["HF_HOME"] = os.path.join(BASE_DIR, ".hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

# Config
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-5
SEED = 42
LABEL2ID = {"normal": 0, "hatespeech": 1, "offensive": 2}
ID2LABEL = {0: "normal", 1: "hatespeech", 2: "offensive"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() ele "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device : {DEVICE}  |  Model : {MODEL_NAME}")

# Load data
posts = pd.read_csv(os.path.join(DATA_DIR, "hatexplain_posts.csv"))
exploded = pd.read_csv(os.path.join(DATA_DIR, "hatexplain_exploded.csv"))

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_df = posts[posts["split"] == "train"].reset_index(drop=True)
val_df = posts[posts["split"] == "val"].reset_index(drop=True)
test_df = posts[posts["split"] == "test"].reset_index(drop=True)

print(f"\nTrain: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = TextDataset(train_df["text"].tolist(), train_df["majority_label"].map(LABEL2ID).tolist(), tokenizer)
val_ds = TextDataset(val_df["text"].tolist(), val_df["majority_label"].map(LABEL2ID).tolist(), tokenizer)
test_ds = TextDataset(test_df["text"].tolist(), test_df["majority_label"].map(LABEL2ID).tolist(), tokenizer)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
).to(DEVICE)

total_steps = len(train_dl) * EPOCHS
warmup_steps = int(0.1 * total_steps)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

history = {"train_loss": [], "val_loss": [], "val_f1": []}
best_val_f1, best_state = 0.0, None

print("\n" + "═"*60)
print(f"  TRAINING — {EPOCHS} epochs | {len(train_dl)} batches/epoch")
print("═"*60)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_dl):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=lbl)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += out.loss.item()

        if (step + 1) % 150 == 0:
            print(f"  Ep {epoch+1} | Step {step+1}/{len(train_dl)} | Loss {running_loss/(step+1):.4f}")

    avg_train = running_loss / len(train_dl)

    model.eval()
    val_loss, preds_v, labels_v = 0.0, [], []
    with torch.no_grad():
        for batch in val_dl:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask, labels=lbl)
            val_loss += out.loss.item()
            preds_v.extend(out.logits.argmax(-1).cpu().numpy())
            labels_v.extend(lbl.cpu().numpy())

    avg_val = val_loss / len(val_dl)
    val_f1 = f1_score(labels_v, preds_v, average="macro")

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["val_f1"].append(val_f1)

    print(f"\n  ── Epoch {epoch+1} ─────────────────────────────────────")
    print(f"     Train Loss : {avg_train:.4f}")
    print(f"     Val   Loss : {avg_val:.4f}")
    print(f"     Val   F1   : {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(OUT_DIR, "bert_best_state.pt"))
        print(f"      Best saved  (F1={best_val_f1:.4f})")

model.load_state_dict(best_state)
model.to(DEVICE)
print(f"\n   Best model loaded  (Val F1={best_val_f1:.4f})")

def run_inference(texts, batch_size=64):
    ds = TextDataset(texts, labels=None, tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds = out.logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
    return all_preds, all_probs

print("\n" + "═"*60)
print("  TEST EVALUATION")
print("═"*60)

test_preds, test_probs = run_inference(test_df["text"].tolist())
test_labels = test_df["majority_label"].map(LABEL2ID).tolist()
test_f1 = f1_score(test_labels, test_preds, average="macro")
test_acc = accuracy_score(test_labels, test_preds)
print(f"\n  Macro-F1 : {test_f1:.4f}  |  Accuracy : {test_acc:.4f}")
print(f"\n{classification_report(test_labels, test_preds, target_names=['normal','hatespeech','offensive'], digits=4)}")

print("  Running inference on all df_posts...")
all_preds_posts, all_probs_posts = run_inference(posts["text"].tolist())

df_posts_bert = posts.copy()
df_posts_bert["bert_pred"] = [ID2LABEL[p] for p in all_preds_posts]
df_posts_bert["bert_pred_int"] = all_preds_posts
df_posts_bert["prob_normal"] = [p[0] for p in all_probs_posts]
df_posts_bert["prob_hatespeech"] = [p[1] for p in all_probs_posts]
df_posts_bert["prob_offensive"] = [p[2] for p in all_probs_posts]
df_posts_bert["correct"] = df_posts_bert["bert_pred"] == df_posts_bert["majority_label"]

print("  Running inference on all df_exploded...")
all_preds_exp, all_probs_exp = run_inference(exploded["text"].tolist())

df_exploded_bert = exploded.copy()
df_exploded_bert["bert_pred"] = [ID2LABEL[p] for p in all_preds_exp]
df_exploded_bert["bert_pred_int"] = all_preds_exp
df_exploded_bert["prob_normal"] = [p[0] for p in all_probs_exp]
df_exploded_bert["prob_hatespeech"] = [p[1] for p in all_probs_exp]
df_exploded_bert["prob_offensive"] = [p[2] for p in all_probs_exp]
df_exploded_bert["correct"] = df_exploded_bert["bert_pred"] == df_exploded_bert["majority_label"]

def export_csv(df, path):
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, list)).any():
            df_out[col] = df_out[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    df_out.to_csv(path, index=False)

export_csv(df_posts_bert, os.path.join(OUT_DIR, "df_posts_bert.csv"))
export_csv(df_exploded_bert, os.path.join(OUT_DIR, "df_exploded_bert.csv"))

print(f"\n   df_posts_bert    → {OUT_DIR}/df_posts_bert.csv")
print(f"   df_exploded_bert → {OUT_DIR}/df_exploded_bert.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"BERT Results — HateXplain  |  Test Macro-F1={test_f1:.4f}  Acc={test_acc:.4f}", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(range(1, EPOCHS+1), history["train_loss"], "o-", label="Train Loss")
ax.plot(range(1, EPOCHS+1), history["val_loss"], "s-", label="Val Loss")
ax.set_title("Loss Curves")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

ax = axes[1]
ax.plot(range(1, EPOCHS+1), history["val_f1"], "D-")
ax.set_ylim(0, 1)
ax.set_title("Val Macro-F1")
ax.set_xlabel("Epoch")

ax = axes[2]
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
            xticklabels=["normal","hate","offensive"],
            yticklabels=["normal","hate","offensive"])
ax.set_title("Confusion Matrix (Test)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bert_results.png"), dpi=150, bbox_inches="tight")
plt.close()

print(" DONE — BERT pipeline complete")
