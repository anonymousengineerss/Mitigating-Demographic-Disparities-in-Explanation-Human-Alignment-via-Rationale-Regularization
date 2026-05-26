!pip install -q transformers==4.40.0 datasets scikit-learn torch accelerate

# @title Complete Adv-Debiased Roberta Set
#  Run this single cell in Colab → downloads everything → exports CSVs


import json, os, ast
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display
from google.colab import files

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 50)

# ── 1. DOWNLOAD 
print("━"*65)
print("  STEP 1 — DOWNLOADING HateXplain FROM GITHUB")
print("━"*65)

DATA_URL  = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
SPLIT_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"

raw    = requests.get(DATA_URL).json()
splits = requests.get(SPLIT_URL).json()

train_ids = set(splits["train"])
val_ids   = set(splits["val"])
test_ids  = set(splits["test"])

print(f"   Posts loaded     : {len(raw):,}")
print(f"   Train / Val / Test : {len(train_ids):,} / {len(val_ids):,} / {len(test_ids):,}")

# ── 2. PARSE INTO FLAT DATAFRAME ─────
print("\n━"*65)
print("  STEP 2 — PARSING ANNOTATIONS")
print("━"*65)

records = []
for post_id, entry in raw.items():
    tokens     = entry["post_tokens"]
    text       = " ".join(tokens)
    rationales = entry.get("rationales", [])

    if   post_id in train_ids: split = "train"
    elif post_id in val_ids:   split = "val"
    elif post_id in test_ids:  split = "test"
    else:                      split = "unknown"

    for i, ann in enumerate(entry.get("annotators", [])):
        label    = ann["label"]            # "hatespeech" / "normal" / "offensive"
        targets  = ann.get("target", [])   # list of identity communities

        rat_mask   = rationales[i] if i < len(rationales) else []
        rat_tokens = [t for t, r in zip(tokens, rat_mask) if r == 1]

        records.append({
            "post_id"            : post_id,
            "split"              : split,
            "text"               : text,
            "tokens"             : tokens,
            "label"              : label,
            "target_groups"      : targets,
            "rationale_mask"     : rat_mask,
            "rationale_tokens"   : rat_tokens,
            "annotator_id"       : ann.get("annotator_id"),
            "n_tokens"           : len(tokens),
            "n_rationale_tokens" : len(rat_tokens),
            "has_rationale"      : len(rat_mask) > 0,
        })

df_hx = pd.DataFrame(records)

# Majority-vote label per post
majority = (
    df_hx.groupby("post_id")["label"]
         .agg(lambda x: x.value_counts().idxmax())
         .reset_index()
         .rename(columns={"label": "majority_label"})
)
df_hx = df_hx.merge(majority, on="post_id", how="left")

# Majority-vote rationale per post (token-level, ≥2/3 annotators)
def majority_rationale(group):
    masks = [m for m in group["rationale_mask"].tolist() if len(m) > 0]
    if not masks: return []
    max_len = max(len(m) for m in masks)
    combined = [sum(m[t] if t < len(m) else 0 for m in masks) for t in range(max_len)]
    return [1 if v >= 2 else 0 for v in combined]

maj_rat = (
    df_hx.groupby("post_id")
         .apply(majority_rationale)
         .reset_index()
         .rename(columns={0: "majority_rationale_mask"})
)
df_hx = df_hx.merge(maj_rat, on="post_id", how="left")
df_hx["majority_rationale_tokens"] = df_hx.apply(
    lambda r: [t for t, v in zip(r["tokens"], r["majority_rationale_mask"]) if v == 1]
              if isinstance(r["majority_rationale_mask"], list) else [],
    axis=1
)

# Exploded version — one row per (annotation × target group), excluding "None"
df_hx_exploded = df_hx.explode("target_groups").reset_index(drop=True)
df_hx_exploded = df_hx_exploded[
    df_hx_exploded["target_groups"].notna() &
    (df_hx_exploded["target_groups"] != "None")
].copy()

print(f"   df_hx shape         : {df_hx.shape}")
print(f"   df_hx_exploded shape : {df_hx_exploded.shape}")

# ── 3. FULL DATASET STATISTICS ───────
print("\n" + "═"*65)
print("  STEP 3 — FULL DATASET STATISTICS")
print("═"*65)

print("\n  ┌─ LABEL DISTRIBUTION (annotation-level) ─")
lbl_counts = df_hx["label"].value_counts()
for lbl, cnt in lbl_counts.items():
    bar = "█" * int(cnt / 500)
    print(f"  │  {lbl:<12} {cnt:>6,}  {bar}")
print("  └─────────────────────┘")

print("\n  ┌─ MAJORITY LABEL DISTRIBUTION (post-level) ──────────────────┐")
maj_counts = df_hx.drop_duplicates("post_id")["majority_label"].value_counts()
for lbl, cnt in maj_counts.items():
    bar = "█" * int(cnt / 300)
    print(f"  │  {lbl:<12} {cnt:>6,}  {bar}")
print("  └─────────────────────┘")

print("\n  ┌─ SPLIT DISTRIBUTION ────────────────────")
split_counts = df_hx.drop_duplicates("post_id")["split"].value_counts()
for sp, cnt in split_counts.items():
    print(f"  │  {sp:<10} {cnt:>6,}")
print("  └─────────────────────┘")

print("\n  ┌─ RATIONALE COVERAGE ────────────────────")
total         = len(df_hx)
has_rat       = df_hx["has_rationale"].sum()
pct           = has_rat / total * 100
avg_rat_tok   = df_hx["n_rationale_tokens"].mean()
avg_all_tok   = df_hx["n_tokens"].mean()
avg_density   = df_hx[df_hx["has_rationale"]]["n_rationale_tokens"].mean() / \
                df_hx[df_hx["has_rationale"]]["n_tokens"].mean()
print(f"  │  Annotations with rationale : {has_rat:>6,} / {total:,} ({pct:.1f}%)")
print(f"  │  Avg rationale tokens       : {avg_rat_tok:.2f}")
print(f"  │  Avg total tokens           : {avg_all_tok:.2f}")
print(f"  │  Avg rationale density (ρ)  : {avg_density:.4f}  ← confirms AUPRC > AUROC")
print("  └─────────────────────┘")

print("\n  ┌─ TARGET GROUP DISTRIBUTION (all annotations, excl. None) ──┐")
tgt = df_hx_exploded["target_groups"].value_counts()
for grp, cnt in tgt.items():
    bar = "█" * int(cnt / 300)
    print(f"  │  {grp:<18} {cnt:>6,}  {bar}")
print("  └─────────────────────┘")

# ── 4. EFG-RELEVANT AXIS SUBSETS (8 primary groups) ──────────────────────────
print("\n" + "═"*65)
print("  STEP 4 — PLAUSIBILITY-EFG AXIS SUBSETS (rationale-bearing only)")
print("═"*65)

PRIMARY_GROUPS = ["African","Islam","Women","Jewish","Homosexual","Refugee","Arab","Caucasian"]

axis_dfs = {}
for grp in PRIMARY_GROUPS:
    sub = df_hx_exploded[
        (df_hx_exploded["target_groups"] == grp) &
        (df_hx_exploded["has_rationale"]  == True)
    ].copy()
    if len(sub) > 0:
        axis_dfs[grp] = sub

print(f"\n  {'Group':<15} {'Total':>7} {'Hatespeech':>12} {'Offensive':>11} "
      f"{'Normal':>8} {'Test rows':>10} {'Avg ρ':>8}")
print("  " + "─"*75)
for grp, sub in axis_dfs.items():
    hs   = (sub["label"] == "hatespeech").sum()
    off  = (sub["label"] == "offensive").sum()
    norm = (sub["label"] == "normal").sum()
    test = (sub["split"] == "test").sum()
    rho  = (sub["n_rationale_tokens"] / sub["n_tokens"]).mean()
    print(f"  {grp:<15} {len(sub):>7,} {hs:>12,} {off:>11,} {norm:>8,} {test:>10,} {rho:>8.4f}")

# ── 5. SAMPLE ENTRIES ────────────────
print("\n" + "═"*65)
print("  STEP 5 — SAMPLE ENTRIES PER GROUP (hatespeech + rationale)")
print("═"*65)

for grp in PRIMARY_GROUPS:
    if grp not in axis_dfs: continue
    sub = axis_dfs[grp][axis_dfs[grp]["label"] == "hatespeech"]
    if len(sub) == 0: continue
    row = sub.iloc[0]
    print(f"\n  [{grp.upper()}]")
    print(f"  Text      : {row['text'][:95]}")
    print(f"  Rationale : {row['rationale_tokens']}")
    print(f"  Split     : {row['split']}  |  Annotator: {row['annotator_id']}")

# ── 6. VISUALISATIONS ────────────────
print("\n" + "═"*65)
print("  STEP 6 — GENERATING VISUALISATIONS")
print("═"*65)

fig = plt.figure(figsize=(18, 12))
fig.suptitle("HateXplain — Dataset Overview for Plausibility-EFG Study",
             fontsize=16, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 6a — Label distribution
ax1 = fig.add_subplot(gs[0, 0])
lbl_counts.plot(kind="bar", ax=ax1, color=["#E8605A","#F0A500","#4C9BE8"], edgecolor="white")
ax1.set_title("Label Distribution\n(annotation-level)", fontweight="bold")
ax1.set_xlabel(""); ax1.set_ylabel("Count")
ax1.set_xticklabels(lbl_counts.index, rotation=0)
for p in ax1.patches:
    ax1.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                 ha="center", va="bottom", fontsize=9)

# 6b — Split distribution
ax2 = fig.add_subplot(gs[0, 1])
split_counts_plot = df_hx.drop_duplicates("post_id")["split"].value_counts()
split_counts_plot.plot(kind="bar", ax=ax2, color=["#6DC5A0","#9B8ED4","#F0A500"], edgecolor="white")
ax2.set_title("Train / Val / Test Split\n(post-level)", fontweight="bold")
ax2.set_xlabel(""); ax2.set_ylabel("Posts")
ax2.set_xticklabels(split_counts_plot.index, rotation=0)
for p in ax2.patches:
    ax2.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                 ha="center", va="bottom", fontsize=9)

# 6c — Rationale density histogram
ax3 = fig.add_subplot(gs[0, 2])
density = df_hx[df_hx["has_rationale"]]["n_rationale_tokens"] / \
          df_hx[df_hx["has_rationale"]]["n_tokens"]
ax3.hist(density, bins=30, color="#4C9BE8", edgecolor="white", alpha=0.85)
ax3.axvline(density.mean(), color="#E8605A", linestyle="--", linewidth=2,
            label=f"Mean ρ = {density.mean():.3f}")
ax3.set_title("Rationale Token Density (ρ)\nacross annotated posts", fontweight="bold")
ax3.set_xlabel("ρ = rationale tokens / total tokens")
ax3.set_ylabel("Frequency")
ax3.legend(fontsize=9)

# 6d — Target group distribution (primary 8)
ax4 = fig.add_subplot(gs[1, 0:2])
grp_sizes  = [len(axis_dfs[g]) for g in PRIMARY_GROUPS if g in axis_dfs]
grp_labels = [g for g in PRIMARY_GROUPS if g in axis_dfs]
colors     = ["#4C9BE8","#6DC5A0","#F0A500","#E8605A","#9B8ED4","#F4A261","#2A9D8F","#E76F51"]
bars = ax4.bar(grp_labels, grp_sizes, color=colors[:len(grp_labels)], edgecolor="white")
ax4.set_title("Plausibility-EFG Axis Groups — Annotation Count (with rationale)",
              fontweight="bold")
ax4.set_xlabel("Identity Target Group")
ax4.set_ylabel("Annotation Rows")
for bar, val in zip(bars, grp_sizes):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f"{val:,}", ha="center", va="bottom", fontsize=9)

# 6e — Rationale coverage by label
ax5 = fig.add_subplot(gs[1, 2])
cov = df_hx.groupby("label")["has_rationale"].mean() * 100
cov.plot(kind="bar", ax=ax5, color=["#E8605A","#4C9BE8","#F0A500"], edgecolor="white")
ax5.set_title("Rationale Coverage\nby Label Class (%)", fontweight="bold")
ax5.set_xlabel(""); ax5.set_ylabel("% Annotations with Rationale")
ax5.set_xticklabels(cov.index, rotation=0)
ax5.set_ylim(0, 105)
for p in ax5.patches:
    ax5.annotate(f"{p.get_height():.1f}%",
                 (p.get_x()+p.get_width()/2, p.get_height()+1),
                 ha="center", va="bottom", fontsize=9)

plt.savefig("hatexplain_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Visualisation saved: hatexplain_overview.png")

# ── 7. EXPORT CSVs

print("\n" + "═"*65)
print("  STEP 7 — EXPORTING CSVS")
print("═"*65)

def serialize(df):
    df = df.copy()
    for col in ["tokens","target_groups","rationale_mask",
                "rationale_tokens","majority_rationale_mask",
                "majority_rationale_tokens"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x)
    return df

os.makedirs("hatexplain_export", exist_ok=True)
os.makedirs("hatexplain_export/groups", exist_ok=True)

# Main annotation-level CSV
serialize(df_hx).to_csv("hatexplain_export/hatexplain_annotations.csv", index=False)

# Exploded per-group CSV
serialize(df_hx_exploded).to_csv("hatexplain_export/hatexplain_exploded.csv", index=False)

# Per-group CSVs (EFG axes — rationale-bearing only)
for grp, sub in axis_dfs.items():
    serialize(sub).to_csv(
        f"hatexplain_export/groups/{grp.lower()}.csv", index=False)

# Post-level deduplicated CSV (one row per post, majority label + majority rationale)
df_post = df_hx.drop_duplicates("post_id")[
    ["post_id","split","text","tokens","majority_label",
     "majority_rationale_mask","majority_rationale_tokens",
     "n_tokens"]
].copy()
serialize(df_post).to_csv("hatexplain_export/hatexplain_posts.csv", index=False)

print(f"   hatexplain_annotations.csv  → {len(df_hx):,} rows")
print(f"   hatexplain_exploded.csv     → {len(df_hx_exploded):,} rows")
print(f"   hatexplain_posts.csv        → {len(df_post):,} rows")
for grp, sub in axis_dfs.items():
    print(f"   groups/{grp.lower()}.csv       → {len(sub):,} rows")

# ── 8. DOWNLOAD EVERYTHING FROM COLAB 
print("\n" + "═"*65)
print("  STEP 8 — DOWNLOADING FILES TO LOCAL")
print("═"*65)

files.download("hatexplain_export/hatexplain_annotations.csv")
files.download("hatexplain_export/hatexplain_exploded.csv")
files.download("hatexplain_export/hatexplain_posts.csv")
files.download("hatexplain_overview.png")
for grp in axis_dfs:
    files.download(f"hatexplain_export/groups/{grp.lower()}.csv")

# ── 9. FINAL SUMMARY ─────────────────
print("\n" + "═"*65)
print("  FINAL SUMMARY")
print("═"*65)
print(f"""
  DATAFRAMES IN MEMORY:
  ┌
  │  df_hx              → {len(df_hx):>6,} rows  (all annotations)          │
  │  df_hx_exploded     → {len(df_hx_exploded):>6,} rows  (annotation × group)     │
  │  df_post            → {len(df_post):>6,} rows  (one per post)            │
  │  axis_dfs           → {len(axis_dfs):>6}  group subsets (8 EFG axes)   │
  

  KEY COLUMNS FOR PLAUSIBILITY-EFG:
  ┌
  │  text                   → raw input for tokenizer           │
  │  majority_label         → ground-truth 3-class label        │
  │  majority_rationale_mask→ binary token mask (≥2/3 votes)   │
  │  target_groups          → EFG stratification axis           │
  │  split                  → train / val / test                │
  │  has_rationale          → False for normal posts (expected) │
  

  LABEL MAP   : hatespeech=1 | offensive=2 | normal=0
  AVG ρ       : {avg_density:.4f}  (confirms AUPRC > AUROC per Lemma 1)
  EFG GROUPS  : {list(axis_dfs.keys())}

   ALL FILES DOWNLOADED — upload to Gilbreth and run sbatch
""")

#   @title Load + Slim HateXplain CSVs — only columns needed for BERT + EFG


import pandas as pd, ast, json

def safe_parse(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except: return []
    return []

# ── Load ───────
posts    = pd.read_csv("hatexplain_export/hatexplain_posts.csv")
exploded = pd.read_csv("hatexplain_export/hatexplain_exploded.csv")

# ── Slim to only what you need ───────
posts = posts[["post_id", "split", "text", "majority_label",
               "majority_rationale_mask"]].copy()

exploded = exploded[["post_id", "split", "text", "label", "majority_label",
                     "target_groups", "majority_rationale_mask",
                     "has_rationale"]].copy()

# ── Parse list columns ───────────────
posts["majority_rationale_mask"]    = posts["majority_rationale_mask"].apply(safe_parse)
exploded["majority_rationale_mask"] = exploded["majority_rationale_mask"].apply(safe_parse)

# ── Filter exploded: valid groups + has rationale only ────────────────────────
PRIMARY_GROUPS = ["African","Islam","Women","Jewish","Homosexual","Refugee","Arab","Caucasian"]

exploded = exploded[
    exploded["target_groups"].isin(PRIMARY_GROUPS) &
    (exploded["has_rationale"] == True)
].reset_index(drop=True)

# ── Show posts ──
print("═"*70)
print("  df_posts  —  ONE ROW PER POST")
print("═"*70)
print(f"  Shape : {posts.shape}")
print(f"\n  Columns:")
for c in posts.columns:
    print(f"   • {c}")
print(f"\n  Label distribution:")
print(posts["majority_label"].value_counts().to_string())
print(f"\n  Split distribution:")
print(posts["split"].value_counts().to_string())
print(f"\n  Rationale coverage:")
has = posts["majority_rationale_mask"].apply(lambda x: sum(x) > 0)
print(f"   Posts with ≥1 rationale token : {has.sum():,} / {len(posts):,} ({has.mean()*100:.1f}%)")
print(f"\n  Sample (5 rows):")
display(posts[["post_id","split","text","majority_label","majority_rationale_mask"]]
        .head(5)
        .assign(majority_rationale_mask=lambda df:
                df["majority_rationale_mask"].apply(lambda x: str(x[:8])+"…" if len(x)>8 else str(x))))

# ── Show exploded─
print("\n" + "═"*70)
print("  df_exploded  —  ONE ROW PER ANNOTATION × TARGET GROUP (EFG axes)")
print("═"*70)
print(f"  Shape : {exploded.shape}")
print(f"\n  Columns:")
for c in exploded.columns:
    print(f"   • {c}")
print(f"\n  Group distribution (EFG axes):")
grp_counts = exploded["target_groups"].value_counts()
for grp, cnt in grp_counts.items():
    hs  = (exploded[exploded["target_groups"]==grp]["label"]=="hatespeech").sum()
    off = (exploded[exploded["target_groups"]==grp]["label"]=="offensive").sum()
    print(f"   {grp:<15} {cnt:>5,} rows  (hate: {hs:,}  off: {off:,})")
print(f"\n  Split breakdown:")
print(exploded["split"].value_counts().to_string())
print(f"\n  Sample (5 rows):")
display(exploded[["post_id","split","text","majority_label","target_groups","majority_rationale_mask"]]
        .head(5)
        .assign(text=lambda df: df["text"].str[:60]+"…",
                majority_rationale_mask=lambda df:
                df["majority_rationale_mask"].apply(lambda x: str(x[:8])+"…" if len(x)>8 else str(x))))

# ── Final check ─
print("\n" + "═"*70)
print("  READY — columns confirmed")
print("═"*70)
print(f"""
  df_posts    → {posts.shape[0]:,} rows
    text                     raw input for tokenizer
    majority_label           hatespeech / offensive / normal
    majority_rationale_mask  binary token list  (AUPRC ground truth)
    split                    train / val / test

  df_exploded → {exploded.shape[0]:,} rows
    text                     raw input for tokenizer
    majority_label           hatespeech / offensive / normal
    majority_rationale_mask  binary token list  (AUPRC ground truth)
    target_groups            EFG subgroup axis  {PRIMARY_GROUPS}
    split                    train / val / test
    has_rationale            all True (already filtered)
""")
!pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os

#   @title Load + Slim HateXplain CSVs — only columns needed for BERT + EFG


import pandas as pd, ast, json

def safe_parse(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except: return []
    return []

# ── Load ───────
posts    = pd.read_csv("hatexplain_export/hatexplain_posts.csv")
exploded = pd.read_csv("hatexplain_export/hatexplain_exploded.csv")

# ── Slim to only what you need ───────
posts = posts[["post_id", "split", "text", "majority_label",
               "majority_rationale_mask"]].copy()

exploded = exploded[["post_id", "split", "text", "label", "majority_label",
                     "target_groups", "majority_rationale_mask",
                     "has_rationale"]].copy()

# ── Parse list columns ───────────────
posts["majority_rationale_mask"]    = posts["majority_rationale_mask"].apply(safe_parse)
exploded["majority_rationale_mask"] = exploded["majority_rationale_mask"].apply(safe_parse)

# ── Filter exploded: valid groups + has rationale only ────────────────────────
PRIMARY_GROUPS = ["African","Islam","Women","Jewish","Homosexual","Refugee","Arab","Caucasian"]

exploded = exploded[
    exploded["target_groups"].isin(PRIMARY_GROUPS) &
    (exploded["has_rationale"] == True)
].reset_index(drop=True)

# ── Show posts ──
print("═"*70)
print("  df_posts  —  ONE ROW PER POST")
print("═"*70)
print(f"  Shape : {posts.shape}")
print(f"\n  Columns:")
for c in posts.columns:
    print(f"   • {c}")
print(f"\n  Label distribution:")
print(posts["majority_label"].value_counts().to_string())
print(f"\n  Split distribution:")
print(posts["split"].value_counts().to_string())
print(f"\n  Rationale coverage:")
has = posts["majority_rationale_mask"].apply(lambda x: sum(x) > 0)
print(f"   Posts with ≥1 rationale token : {has.sum():,} / {len(posts):,} ({has.mean()*100:.1f}%)")
print(f"\n  Sample (5 rows):")
display(posts[["post_id","split","text","majority_label","majority_rationale_mask"]]
        .head(5)
        .assign(majority_rationale_mask=lambda df:
                df["majority_rationale_mask"].apply(lambda x: str(x[:8])+"…" if len(x)>8 else str(x))))

# ── Show exploded─
print("\n" + "═"*70)
print("  df_exploded  —  ONE ROW PER ANNOTATION × TARGET GROUP (EFG axes)")
print("═"*70)
print(f"  Shape : {exploded.shape}")
print(f"\n  Columns:")
for c in exploded.columns:
    print(f"   • {c}")
print(f"\n  Group distribution (EFG axes):")
grp_counts = exploded["target_groups"].value_counts()
for grp, cnt in grp_counts.items():
    hs  = (exploded[exploded["target_groups"]==grp]["label"]=="hatespeech").sum()
    off = (exploded[exploded["target_groups"]==grp]["label"]=="offensive").sum()
    print(f"   {grp:<15} {cnt:>5,} rows  (hate: {hs:,}  off: {off:,})")
print(f"\n  Split breakdown:")
print(exploded["split"].value_counts().to_string())
print(f"\n  Sample (5 rows):")
display(exploded[["post_id","split","text","majority_label","target_groups","majority_rationale_mask"]]
        .head(5)
        .assign(text=lambda df: df["text"].str[:60]+"…",
                majority_rationale_mask=lambda df:
                df["majority_rationale_mask"].apply(lambda x: str(x[:8])+"…" if len(x)>8 else str(x))))

# ── Final check ─
print("\n" + "═"*70)
print("  READY — columns confirmed")
print("═"*70)
print(f"""
  df_posts    → {posts.shape[0]:,} rows
    text                     raw input for tokenizer
    majority_label           hatespeech / offensive / normal
    majority_rationale_mask  binary token list  (AUPRC ground truth)
    split                    train / val / test

  df_exploded → {exploded.shape[0]:,} rows
    text                     raw input for tokenizer
    majority_label           hatespeech / offensive / normal
    majority_rationale_mask  binary token list  (AUPRC ground truth)
    target_groups            EFG subgroup axis  {PRIMARY_GROUPS}
    split                    train / val / test
    has_rationale            all True (already filtered)
""")
df_posts    = posts.copy()
df_exploded = exploded.copy()
print("df_posts" in globals(), "df_exploded" in globals())
print(type(df_posts), type(df_exploded))
print(df_exploded.shape)
import os, random, json, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

df_posts = posts.copy()
df_exploded = exploded.copy()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}  |  Model : roberta-base-ADV")

TEXT_COL = "text"
LABEL_COL = "majority_label"
GROUP_COL = "target_groups"

label2id = {"normal": 0, "hatespeech": 1, "offensive": 2}
id2label = {v: k for k, v in label2id.items()}

df_posts[TEXT_COL] = df_posts[TEXT_COL].astype(str).fillna("")
df_exploded[TEXT_COL] = df_exploded[TEXT_COL].astype(str).fillna("")
df_posts[LABEL_COL] = df_posts[LABEL_COL].astype(str).str.strip().str.lower()
df_exploded[LABEL_COL] = df_exploded[LABEL_COL].astype(str).str.strip().str.lower()
df_exploded[GROUP_COL] = df_exploded[GROUP_COL].astype(str).str.strip()

group_names = sorted(df_exploded[GROUP_COL].dropna().unique().tolist())
group2id = {g: i for i, g in enumerate(group_names)}
id2group = {v: k for k, v in group2id.items()}

df_posts["label_id"] = df_posts[LABEL_COL].map(label2id).astype(int)
df_exploded["label_id"] = df_exploded[LABEL_COL].map(label2id).astype(int)
df_exploded["group_id"] = df_exploded[GROUP_COL].map(group2id).astype(int)

train_df = df_exploded[df_exploded["split"].eq("train")].copy()
val_df = df_exploded[df_exploded["split"].eq("val")].copy()
test_df = df_exploded[df_exploded["split"].eq("test")].copy()

print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-5
PATIENCE = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class AdvDataset(Dataset):
    def __init__(self, df, max_len=128):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = tokenizer(
            str(row[TEXT_COL]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "groups": torch.tensor(int(row["group_id"]), dtype=torch.long),
        }

train_loader = DataLoader(AdvDataset(train_df, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(AdvDataset(val_df, MAX_LEN), batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(AdvDataset(test_df, MAX_LEN), batch_size=64, shuffle=False, num_workers=0)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class RobertaAdv(nn.Module):
    def __init__(self, base_model, num_labels, num_groups, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.task_head = nn.Linear(hidden, num_labels)
        self.adv_head = nn.Linear(hidden, num_groups)

    def forward(self, input_ids, attention_mask, labels=None, groups=None, lambd_adv=0.5):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0])
        task_logits = self.task_head(cls)
        adv_logits = self.adv_head(grad_reverse(cls, lambd_adv))
        loss = None
        if labels is not None and groups is not None:
            task_loss = nn.CrossEntropyLoss()(task_logits, labels)
            adv_loss = nn.CrossEntropyLoss()(adv_logits, groups)
            loss = task_loss + adv_loss
        return {"loss": loss, "task_logits": task_logits, "adv_logits": adv_logits}

model = RobertaAdv(MODEL_NAME, num_labels=3, num_groups=len(group2id)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(0.1 * total_steps)),
    num_training_steps=total_steps
)

def evaluate(loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"], lambd_adv=0.0)
            pred = out["task_logits"].argmax(-1)
            ys.extend(batch["labels"].cpu().tolist())
            ps.extend(pred.cpu().tolist())
    return f1_score(ys, ps, average="macro"), accuracy_score(ys, ps)

best_f1 = -1
best_state = None
bad_epochs = 0
history = {"train_loss": [], "val_f1": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(
            batch["input_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
            groups=batch["groups"],
            lambd_adv=0.5
        )
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += loss.item()
        if step % 150 == 0 or step == len(train_loader):
            print(f"Ep {epoch} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

    train_loss = running / len(train_loader)
    val_f1, val_acc = evaluate(val_loader)
    history["train_loss"].append(train_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    print(f"\n  ── Epoch {epoch} ────────")
    print(f"     Train Loss : {train_loss:.4f}")
    print(f"     Val F1     : {val_f1:.4f}")
    print(f"     Val Acc    : {val_acc:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
        print(f"      Best saved  (F1={best_f1:.4f})")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"     ⏹ Early stop triggered (patience={PATIENCE})")
            break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"\n Best model loaded  (Val F1={best_f1:.4f})")

test_f1, test_acc = evaluate(test_loader)

print(f"\n  Macro-F1 : {test_f1:.4f}  |  Accuracy : {test_acc:.4f}\n")

y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"], lambd_adv=0.0)
        pred = out["task_logits"].argmax(-1)
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=["normal", "hatespeech", "offensive"],
    digits=4,
    zero_division=0
))

def run_inference(texts, batch_size=64):
    ds = TextOnlyDataset(texts, tokenizer, MAX_LEN)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(ids, mask, lambd_adv=0.0)
            probs = torch.softmax(out["task_logits"], dim=-1).cpu().numpy()
            preds = out["task_logits"].argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
    return all_preds, all_probs

class TextOnlyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

print("\n  Running inference on all df_posts...")
all_preds_posts, all_probs_posts = run_inference(df_posts["text"].tolist())

df_posts_roberta_adv = df_posts.copy()
df_posts_roberta_adv["roberta_adv_pred_int"] = all_preds_posts
df_posts_roberta_adv["roberta_adv_pred"] = [id2label[p] for p in all_preds_posts]
df_posts_roberta_adv["prob_normal"] = [p[0] for p in all_probs_posts]
df_posts_roberta_adv["prob_hatespeech"] = [p[1] for p in all_probs_posts]
df_posts_roberta_adv["prob_offensive"] = [p[2] for p in all_probs_posts]
df_posts_roberta_adv["correct"] = (df_posts_roberta_adv["roberta_adv_pred"] == df_posts_roberta_adv["majority_label"])

print("  Running inference on all df_exploded...")
all_preds_exp, all_probs_exp = run_inference(df_exploded["text"].tolist())

df_exploded_roberta_adv = df_exploded.copy()
df_exploded_roberta_adv["roberta_adv_pred_int"] = all_preds_exp
df_exploded_roberta_adv["roberta_adv_pred"] = [id2label[p] for p in all_preds_exp]
df_exploded_roberta_adv["prob_normal"] = [p[0] for p in all_probs_exp]
df_exploded_roberta_adv["prob_hatespeech"] = [p[1] for p in all_probs_exp]
df_exploded_roberta_adv["prob_offensive"] = [p[2] for p in all_probs_exp]
df_exploded_roberta_adv["correct"] = (df_exploded_roberta_adv["roberta_adv_pred"] == df_exploded_roberta_adv["majority_label"])


print(f"  Shape : {df_posts_roberta_adv.shape}")
print(df_posts_roberta_adv["roberta_adv_pred"].value_counts().to_string())

print(f"  Shape : {df_exploded_roberta_adv.shape}")
print(df_exploded_roberta_adv["roberta_adv_pred"].value_counts().to_string())

os.makedirs("roberta_adv_output", exist_ok=True)

def export_csv(df, path):
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, list)).any():
            df_out[col] = df_out[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    df_out.to_csv(path, index=False)

export_csv(df_posts_roberta_adv, "roberta_adv_output/df_posts_roberta_adv.csv")
export_csv(df_exploded_roberta_adv, "roberta_adv_output/df_exploded_roberta_adv.csv")

print(f"\n df_posts_roberta_adv    → roberta_adv_output/df_posts_roberta_adv.csv    ({len(df_posts_roberta_adv):,} rows)")
print(f" df_exploded_roberta_adv → roberta_adv_output/df_exploded_roberta_adv.csv ({len(df_exploded_roberta_adv):,} rows)")
print("\n DONE — RoBERTa-ADV pipeline complete")
!pip install -q captum

import os, json, ast, warnings
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from scipy.stats import kruskal
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRIMARY = ["African","Islam","Women","Jewish","Homosexual","Refugee","Arab","Caucasian"]
L2I = {"normal": 0, "hatespeech": 1, "offensive": 2}
I2L = {0: "normal", 1: "hatespeech", 2: "offensive"}

def safe_parse(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []

def get_ig_adv(text, pred_int):
    model.eval()
    enc = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        emb = model.encoder.embeddings.word_embeddings(ids)

    base = torch.zeros_like(emb)

    def fwd(e, m):
        out = model.encoder(inputs_embeds=e, attention_mask=m)
        cls = model.drop(out.last_hidden_state[:, 0])
        logits = model.task_head(cls)
        return logits

    ig = IntegratedGradients(fwd)
    attr = ig.attribute(
        emb,
        base,
        target=pred_int,
        additional_forward_args=(mask,),
        n_steps=50,
        internal_batch_size=10
    )

    scores = attr.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().cpu().numpy())
    real = int(mask.squeeze().sum().item())
    return scores[1:real-1], tokens[1:real-1]

def auprc(sc, rat):
    m = np.array(rat[:len(sc)])
    s = np.array(sc[:len(rat)])
    if len(m) == 0 or m.sum() == 0 or m.sum() == len(m):
        return np.nan
    return float(average_precision_score(m, s))

def iou_f1(sc, rat):
    m = np.array(rat[:len(sc)])
    s = np.array(sc[:len(rat)])
    if len(m) == 0 or m.sum() == 0:
        return np.nan
    pb = (s >= np.median(s)).astype(int)
    inter = (pb & m.astype(int)).sum()
    union = (pb | m.astype(int)).sum()
    return float(inter / union) if union > 0 else 0.0

def run_ig_adv(df_in, name):
    df = df_in.copy().reset_index(drop=True)
    df["majority_rationale_mask"] = df["majority_rationale_mask"].apply(safe_parse)
    igs, toks, auprcs_col, ioufs = [], [], [], []
    errs = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"IG {name}"):
        rat = row["majority_rationale_mask"]
        if not (isinstance(rat, list) and sum(rat) > 0):
            igs.append([])
            toks.append([])
            auprcs_col.append(np.nan)
            ioufs.append(np.nan)
            continue
        try:
            sc, tk = get_ig_adv(row["text"], L2I.get(row["roberta_adv_pred"], 1))
            igs.append(sc.tolist())
            toks.append(tk)
            auprcs_col.append(auprc(sc, rat))
            ioufs.append(iou_f1(sc, rat))
        except Exception:
            errs += 1
            igs.append([])
            toks.append([])
            auprcs_col.append(np.nan)
            ioufs.append(np.nan)
    df["ig_scores"] = igs
    df["ig_tokens"] = toks
    df["auprc"] = auprcs_col
    df["iou_f1"] = ioufs
    print(f"   {name}: {df['auprc'].notna().sum():,} scored | {errs} errors | {df['auprc'].isna().sum():,} NaN (no rationale)")
    return df

def full_numerics_adv(df, split_col="split"):
    print(f"\n  ── Classification Metrics ──────")
    valid = df[df["majority_label"].notna() & df["roberta_adv_pred"].notna()]
    y_true = valid["majority_label"].map(L2I).values
    y_pred = valid["roberta_adv_pred"].map(L2I).values
    print(f"  Overall  Acc={accuracy_score(y_true, y_pred):.4f}  Macro-F1={f1_score(y_true, y_pred, average='macro'):.4f}")
    for sp in ["train","val","test"]:
        sub = valid[valid[split_col] == sp]
        if len(sub) == 0:
            continue
        yt = sub["majority_label"].map(L2I).values
        yp = sub["roberta_adv_pred"].map(L2I).values
        print(f"  {sp:<6}  Acc={accuracy_score(yt, yp):.4f}  Macro-F1={f1_score(yt, yp, average='macro'):.4f}  n={len(sub):,}")

    print(f"\n  ── Plausibility Metrics (IG rows only) ──────────────────────")
    ig_rows = df[df["auprc"].notna()]
    print(f"  Rows with IG scores : {len(ig_rows):,} / {len(df):,}")
    print(f"  Mean AUPRC          : {ig_rows['auprc'].mean():.4f}  std={ig_rows['auprc'].std():.4f}")
    print(f"  Mean IOU-F1         : {ig_rows['iou_f1'].mean():.4f}  std={ig_rows['iou_f1'].std():.4f}")
    for sp in ["train","val","test"]:
        sub = ig_rows[ig_rows[split_col] == sp]
        if len(sub) == 0:
            continue
        print(f"  {sp:<6}  AUPRC={sub['auprc'].mean():.4f}  IOU-F1={sub['iou_f1'].mean():.4f}  n={len(sub):,}")

def efg_table_adv(df, group_col="target_groups"):
    gs, ga = {}, {}
    for g in PRIMARY:
        sub = df[(df[group_col] == g) & df["auprc"].notna()]
        if len(sub) < 10:
            continue
        a = sub["auprc"].values
        b = [np.mean(np.random.choice(a, len(a), replace=True)) for _ in range(1000)]
        gs[g] = {
            "n": len(sub),
            "auprc_mean": float(np.mean(a)),
            "auprc_std": float(np.std(a)),
            "ci_lo": float(np.percentile(b, 2.5)),
            "ci_hi": float(np.percentile(b, 97.5)),
            "iou_f1_mean": float(sub["iou_f1"].dropna().mean())
        }
        ga[g] = a

    means = {g: v["auprc_mean"] for g, v in gs.items()}
    gb = max(means, key=means.get)
    gw = min(means, key=means.get)
    EFG = means[gb] - means[gw]
    h, p = kruskal(*[ga[g] for g in gs])
    a1, a2 = ga[gb], ga[gw]
    pool = np.sqrt((np.std(a1)**2 + np.std(a2)**2) / 2)
    cd = abs(np.mean(a1) - np.mean(a2)) / pool if pool > 0 else 0.0

    print(f"\n  {'Group':<15}{'AUPRC':>8}{'±std':>7}  {'95% CI':<20}{'IOU-F1':>8}{'n':>7}")
    print("  " + "─"*68)
    for g, v in sorted(gs.items(), key=lambda x: -x[1]["auprc_mean"]):
        flag = " ◄ best" if g == gb else (" ◄ worst" if g == gw else "")
        print(f"  {g:<15}{v['auprc_mean']:>8.4f}{v['auprc_std']:>7.4f}  [{v['ci_lo']:.4f},{v['ci_hi']:.4f}]  {v['iou_f1_mean']:>8.4f}{v['n']:>7}{flag}")

    print(f"\n  Plausibility-EFG : {EFG:.4f}")
    print(f"  Best  group      : {gb} ({means[gb]:.4f})")
    print(f"  Worst group      : {gw} ({means[gw]:.4f})")
    print(f"  Cohen's d        : {cd:.4f}")
    print(f"  Kruskal-Wallis p : {p:.4e}  {' significant' if p < 0.01 else '⚠️ not sig'}")
    return gs, EFG, gb, gw, cd, p

print("═"*65)
print("  STEP 1/2  —  IG on df_posts_roberta_adv")
print("═"*65)
df_posts_roberta_adv_ig = run_ig_adv(df_posts_roberta_adv, "posts_adv")

print("\n" + "═"*65)
print("  STEP 2/2  —  IG on df_exploded_roberta_adv")
print("═"*65)
df_exploded_roberta_adv_ig = run_ig_adv(df_exploded_roberta_adv, "exploded_adv")

print("\n" + "═"*65)
print("  COMPLETE NUMERICS — df_posts_roberta_adv_ig")
print("═"*65)
full_numerics_adv(df_posts_roberta_adv_ig)

print("\n" + "═"*65)
print("  COMPLETE NUMERICS — df_exploded_roberta_adv_ig")
print("═"*65)
full_numerics_adv(df_exploded_roberta_adv_ig)

print("\n" + "═"*65)
print("  PLAUSIBILITY-EFG — ALL SPLITS")
print("═"*65)
gs_all, EFG_all, gb_all, gw_all, cd_all, p_all = efg_table_adv(df_exploded_roberta_adv_ig)

print("\n" + "═"*65)
print("  PLAUSIBILITY-EFG — TEST SPLIT ONLY (paper-reported)")
print("═"*65)
gs_test, EFG_test, gb_test, gw_test, cd_test, p_test = efg_table_adv(
    df_exploded_roberta_adv_ig[df_exploded_roberta_adv_ig["split"] == "test"]
)

os.makedirs("ig_output_roberta_adv", exist_ok=True)

def save_df(df, path):
    d = df.copy()
    for c in ["ig_scores", "ig_tokens", "majority_rationale_mask"]:
        if c in d.columns:
            d[c] = d[c].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    d.to_csv(path, index=False)
    print(f"   {path}  ({len(d):,} rows)")

save_df(df_posts_roberta_adv_ig, "ig_output_roberta_adv/df_posts_roberta_adv_ig.csv")
save_df(df_exploded_roberta_adv_ig, "ig_output_roberta_adv/df_exploded_roberta_adv_ig.csv")

pd.DataFrame([{"group": g, **v} for g, v in gs_test.items()]).to_csv(
    "ig_output_roberta_adv/efg_test_per_group_roberta_adv.csv", index=False
)
pd.DataFrame([{"group": g, **v} for g, v in gs_all.items()]).to_csv(
    "ig_output_roberta_adv/efg_all_per_group_roberta_adv.csv", index=False
)

with open("ig_output_roberta_adv/efg_summary_roberta_adv.json", "w") as f:
    json.dump({
        "model": "roberta-base-ADV",
        "method": "IntegratedGradients",
        "test": {"EFG": EFG_test, "best": gb_test, "worst": gw_test, "cohens_d": cd_test, "kw_p": float(p_test), "per_group": gs_test},
        "all": {"EFG": EFG_all, "best": gb_all, "worst": gw_all, "cohens_d": cd_all, "kw_p": float(p_all), "per_group": gs_all}
    }, f, indent=2)

print(f"""
 COMPLETE — RoBERTa-ADV + Integrated Gradients
   df_posts_roberta_adv_ig    → {len(df_posts_roberta_adv_ig):,} rows
   df_exploded_roberta_adv_ig → {len(df_exploded_roberta_adv_ig):,} rows

   PAPER NUMBERS (test split):
   Plausibility-EFG = {EFG_test:.4f}
   Cohen's d        = {cd_test:.4f}
   KW p             = {p_test:.4e}
   Best group       = {gb_test}
   Worst group      = {gw_test}
""")
#@title shap-  Adv-Debiased Roberta
#@title shap -roberta-adv
!pip install -q shap

import os, json, ast, warnings, random
import numpy as np
import pandas as pd
import torch
import shap
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from scipy.stats import kruskal
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRIMARY = ["African","Islam","Women","Jewish","Homosexual","Refugee","Arab","Caucasian"]
L2I = {"normal": 0, "hatespeech": 1, "offensive": 2}
I2L = {0: "normal", 1: "hatespeech", 2: "offensive"}

df_posts_roberta_adv = df_posts_roberta_adv.copy()
df_exploded_roberta_adv = df_exploded_roberta_adv.copy()

def safe_parse(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []

def predict_fn(texts):
    # SHAP often passes numpy arrays; ensure we have a list of strings
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    else:
        texts = list(texts)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            lambd_adv=0.0
        )
        logits = out["task_logits"]
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()

masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_fn, masker)

def extract_shap_scores(text, pred_int, max_evals=256):
    sv = explainer([text], max_evals=max_evals)
    vals = sv.values[0]
    if vals.ndim == 2:
        vals = vals[:, pred_int]
    tokens = sv.data[0]
    scores = np.maximum(np.array(vals), 0)

    clean_tokens, clean_scores = [], []
    for t, s in zip(tokens, scores):
        if t in ["<s>", "</s>", "<pad>"]:
            continue
        clean_tokens.append(t)
        clean_scores.append(float(s))

    return np.array(clean_scores), clean_tokens

def auprc(sc, rat):
    m = np.array(rat[:len(sc)])
    s = np.array(sc[:len(rat)])
    if len(m) == 0 or m.sum() == 0 or m.sum() == len(m):
        return np.nan
    return float(average_precision_score(m, s))

def iou_f1(sc, rat):
    m = np.array(rat[:len(sc)])
    s = np.array(sc[:len(rat)])
    if len(m) == 0 or m.sum() == 0:
        return np.nan
    pb = (s >= np.median(s)).astype(int)
    inter = (pb & m.astype(int)).sum()
    union = (pb | m.astype(int)).sum()
    return float(inter / union) if union > 0 else 0.0

def sample_balanced_test(df, total_n=1000):
    df = df[df["split"] == "test"].copy()
    if "target_groups" not in df.columns:
        if len(df) > total_n:
            df = df.sample(total_n, random_state=42)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    per_group = max(1, total_n // len(PRIMARY))
    parts = []
    for g in PRIMARY:
        sub = df[df["target_groups"] == g]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(min(per_group, len(sub)), random_state=42))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > total_n:
        out = out.sample(total_n, random_state=42).reset_index(drop=True)
    return out.sample(frac=1, random_state=42).reset_index(drop=True)

def run_shap_roberta_adv(df_in, name, sample_n=1000):
    df = df_in.copy().reset_index(drop=True)
    df["majority_rationale_mask"] = df["majority_rationale_mask"].apply(safe_parse)
    df = sample_balanced_test(df, total_n=sample_n)

    shap_scores_col, shap_tokens_col, auprcs_col, ioufs_col = [], [], [], []
    errs = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"SHAP {name}"):
        rat = row["majority_rationale_mask"]
        if not (isinstance(rat, list) and sum(rat) > 0):
            shap_scores_col.append([])
            shap_tokens_col.append([])
            auprcs_col.append(np.nan)
            ioufs_col.append(np.nan)
            continue
        try:
            pred_int = L2I.get(row["roberta_adv_pred"], 1)
            sc, tk = extract_shap_scores(row["text"], pred_int, max_evals=256)
            shap_scores_col.append(sc.tolist())
            shap_tokens_col.append(tk)
            auprcs_col.append(auprc(sc, rat))
            ioufs_col.append(iou_f1(sc, rat))
        except Exception as e:
            if errs == 0:
                print(f"\n[SHAP ERROR on first failure]: {e}")
            errs += 1
            shap_scores_col.append([])
            shap_tokens_col.append([])
            auprcs_col.append(np.nan)
            ioufs_col.append(np.nan)

    df["shap_scores"] = shap_scores_col
    df["shap_tokens"] = shap_tokens_col
    df["auprc"] = auprcs_col
    df["iou_f1"] = ioufs_col

    print(f"   {name}: {df['auprc'].notna().sum():,} scored | {errs} errors | {df['auprc'].isna().sum():,} NaN (no rationale)")
    return df

def full_numerics_roberta_adv_shap(df, split_col="split"):
    print(f"\n  ── Classification Metrics ──────")
    valid = df[df["majority_label"].notna() & df["roberta_adv_pred"].notna()]
    y_true = valid["majority_label"].map(L2I).values
    y_pred = valid["roberta_adv_pred"].map(L2I).values
    print(f"  Overall  Acc={accuracy_score(y_true, y_pred):.4f}  Macro-F1={f1_score(y_true, y_pred, average='macro'):.4f}")

    for sp in ["train", "val", "test"]:
        sub = valid[valid[split_col] == sp]
        if len(sub) == 0:
            continue
        yt = sub["majority_label"].map(L2I).values
        yp = sub["roberta_adv_pred"].map(L2I).values
        print(f"  {sp:<6}  Acc={accuracy_score(yt, yp):.4f}  Macro-F1={f1_score(yt, yp, average='macro'):.4f}  n={len(sub):,}")

    print(f"\n  ── Plausibility Metrics (SHAP rows only) ────────────────────")
    rows = df[df["auprc"].notna()]
    print(f"  Rows with SHAP scores : {len(rows):,} / {len(df):,}")
    print(f"  Mean AUPRC            : {rows['auprc'].mean():.4f}  std={rows['auprc'].std():.4f}")
    print(f"  Mean IOU-F1           : {rows['iou_f1'].mean():.4f}  std={rows['iou_f1'].std():.4f}")
    for sp in ["train", "val", "test"]:
        sub = rows[rows[split_col] == sp]
        if len(sub) == 0:
            continue
        print(f"  {sp:<6}  AUPRC={sub['auprc'].mean():.4f}  IOU-F1={sub['iou_f1'].mean():.4f}  n={len(sub):,}")

def efg_table_roberta_adv_shap(df, group_col="target_groups"):
    gs, ga = {}, {}
    for g in PRIMARY:
        if group_col not in df.columns:
            continue
        sub = df[(df[group_col] == g) & df["auprc"].notna()]
        if len(sub) < 10:
            continue
        a = sub["auprc"].values
        b = [np.mean(np.random.choice(a, len(a), replace=True)) for _ in range(1000)]
        gs[g] = {
            "n": len(sub),
            "auprc_mean": float(np.mean(a)),
            "auprc_std": float(np.std(a)),
            "ci_lo": float(np.percentile(b, 2.5)),
            "ci_hi": float(np.percentile(b, 97.5)),
            "iou_f1_mean": float(sub["iou_f1"].dropna().mean())
        }
        ga[g] = a

    if not gs:
        return {}, 0.0, None, None, 0.0, np.nan

    means = {g: v["auprc_mean"] for g, v in gs.items()}
    gb = max(means, key=means.get)
    gw = min(means, key=means.get)
    EFG = means[gb] - means[gw]
    h, p = kruskal(*[ga[g] for g in gs]) if len(gs) >= 2 else (np.nan, np.nan)
    a1, a2 = ga[gb], ga[gw]
    pool = np.sqrt((np.std(a1)**2 + np.std(a2)**2) / 2)
    cd = abs(np.mean(a1) - np.mean(a2)) / pool if pool > 0 else 0.0

    print(f"\n  {'Group':<15}{'AUPRC':>8}{'±std':>7}  {'95% CI':<20}{'IOU-F1':>8}{'n':>7}")
    print("  " + "─"*68)
    for g, v in sorted(gs.items(), key=lambda x: -x[1]["auprc_mean"]):
        flag = " ◄ best" if g == gb else (" ◄ worst" if g == gw else "")
        print(f"  {g:<15}{v['auprc_mean']:>8.4f}{v['auprc_std']:>7.4f}  [{v['ci_lo']:.4f},{v['ci_hi']:.4f}]  {v['iou_f1_mean']:>8.4f}{v['n']:>7}{flag}")

    print(f"\n  Plausibility-EFG : {EFG:.4f}")
    print(f"  Best  group      : {gb} ({means[gb]:.4f})")
    print(f"  Worst group      : {gw} ({means[gw]:.4f})")
    print(f"  Cohen's d        : {cd:.4f}")
    print(f"  Kruskal-Wallis p : {p:.4e}  {' significant' if p < 0.01 else '⚠️ not sig'}")
    return gs, EFG, gb, gw, cd, p

print("═"*65)
print("  STEP 1/2  —  SHAP on df_posts_roberta_adv")
print("═"*65)
df_posts_roberta_adv_shap = run_shap_roberta_adv(df_posts_roberta_adv, "posts_roberta_adv", sample_n=1000)

print("\n" + "═"*65)
print("  STEP 2/2  —  SHAP on df_exploded_roberta_adv")
print("═"*65)
df_exploded_roberta_adv_shap = run_shap_roberta_adv(df_exploded_roberta_adv, "exploded_roberta_adv", sample_n=1000)

print("\n" + "═"*65)
print("  COMPLETE NUMERICS — df_posts_roberta_adv_shap")
print("═"*65)
full_numerics_roberta_adv_shap(df_posts_roberta_adv_shap)

print("\n" + "═"*65)
print("  COMPLETE NUMERICS — df_exploded_roberta_adv_shap")
print("═"*65)
full_numerics_roberta_adv_shap(df_exploded_roberta_adv_shap)

print("\n" + "═"*65)
print("  PLAUSIBILITY-EFG — ALL SPLITS")
print("═"*65)
gs_all, EFG_all, gb_all, gw_all, cd_all, p_all = efg_table_roberta_adv_shap(df_exploded_roberta_adv_shap)

print("\n" + "═"*65)
print("  PLAUSIBILITY-EFG — TEST SPLIT ONLY (paper-reported)")
print("═"*65)
gs_test, EFG_test, gb_test, gw_test, cd_test, p_test = efg_table_roberta_adv_shap(
    df_exploded_roberta_adv_shap[df_exploded_roberta_adv_shap["split"] == "test"]
)

os.makedirs("shap_output_roberta_adv", exist_ok=True)

def save_df(df, path):
    d = df.copy()
    for c in ["shap_scores", "shap_tokens", "majority_rationale_mask"]:
        if c in d.columns:
            d[c] = d[c].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    d.to_csv(path, index=False)
    print(f"   {path}  ({len(d):,} rows)")

save_df(df_posts_roberta_adv_shap, "shap_output_roberta_adv/df_posts_roberta_adv_shap.csv")
save_df(df_exploded_roberta_adv_shap, "shap_output_roberta_adv/df_exploded_roberta_adv_shap.csv")

pd.DataFrame([{"group": g, **v} for g, v in gs_test.items()]).to_csv(
    "shap_output_roberta_adv/efg_test_per_group_roberta_adv_shap.csv", index=False
)
pd.DataFrame([{"group": g, **v} for g, v in gs_all.items()]).to_csv(
    "shap_output_roberta_adv/efg_all_per_group_roberta_adv_shap.csv", index=False
)

with open("shap_output_roberta_adv/efg_summary_roberta_adv_shap.json", "w") as f:
    json.dump({
        "model": "roberta-base-ADV",
        "method": "SHAP",
        "test": {
            "EFG": EFG_test,
            "best": gb_test,
            "worst": gw_test,
            "cohens_d": cd_test,
            "kw_p": float(p_test) if p_test is not None else None,
            "per_group": gs_test
        },
        "all": {
            "EFG": EFG_all,
            "best": gb_all,
            "worst": gw_all,
            "cohens_d": cd_all,
            "kw_p": float(p_all) if p_all is not None else None,
            "per_group": gs_all
        }
    }, f, indent=2)

print(f"""
 COMPLETE — RoBERTa-ADV + SHAP
   df_posts_roberta_adv_shap    → {len(df_posts_roberta_adv_shap):,} rows
   df_exploded_roberta_adv_shap → {len(df_exploded_roberta_adv_shap):,} rows

   PAPER NUMBERS (test split):
   Plausibility-EFG = {EFG_test:.4f}
   Cohen's d        = {cd_test:.4f}
   KW p             = {p_test}
   Best group       = {gb_test}
   Worst group      = {gw_test}
""")
#@title mitigation-ablation -Adv-Debiased Roberta
#@title mitigation-ablation -roberta-adv
import os, ast, json, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TEXT_COL = "text"
LABEL_COL = "majority_label"
GROUP_COL = "target_groups"
MAX_LEN = 128

L2I = {"normal": 0, "hatespeech": 1, "offensive": 2}
PRIMARY = ["African", "Islam", "Women", "Jewish", "Homosexual", "Refugee", "Arab", "Caucasian"]
group2id = {g: i for i, g in enumerate(PRIMARY)}

df_train = df_exploded_roberta_adv[df_exploded_roberta_adv["split"] == "train"].copy()
df_val = df_exploded_roberta_adv[df_exploded_roberta_adv["split"] == "val"].copy()
df_test = df_exploded_roberta_adv[df_exploded_roberta_adv["split"] == "test"].copy()

def safe_parse(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []

for d in [df_train, df_val, df_test]:
    d["majority_rationale_mask"] = d["majority_rationale_mask"].apply(safe_parse)
    d["has_rationale"] = d["majority_rationale_mask"].apply(lambda x: isinstance(x, list) and sum(x) > 0)

df_train = df_train[df_train["has_rationale"]].copy()
df_val = df_val[df_val["has_rationale"]].copy()
df_test = df_test[df_test["has_rationale"]].copy()

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)
if not getattr(tokenizer, "is_fast", False):
    raise ValueError("tokenizer must be a fast tokenizer to support word_ids() alignment.")

def align_rationale_mask(text, word_mask, tokenizer, max_len=128):
    words = str(text).split()
    word_mask = list(word_mask) if isinstance(word_mask, list) else []
    if len(word_mask) < len(words):
        word_mask = word_mask + [0] * (len(words) - len(word_mask))
    else:
        word_mask = word_mask[:len(words)]

    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    word_ids = enc.word_ids(batch_index=0)
    aligned = []
    for wi in word_ids:
        if wi is None:
            aligned.append(0.0)
        else:
            aligned.append(float(word_mask[wi]))
    return enc, torch.tensor(aligned, dtype=torch.float)

class FairDataset(Dataset):
    def __init__(self, df, max_len=128):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rat = row["majority_rationale_mask"]
        enc, rat_aligned = align_rationale_mask(row[TEXT_COL], rat, tokenizer, max_len=self.max_len)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(L2I[row[LABEL_COL]], dtype=torch.long),
            "groups": torch.tensor(group2id[row[GROUP_COL]], dtype=torch.long),
            "rat_mask": rat_aligned
        }

train_loader = DataLoader(FairDataset(df_train, MAX_LEN), batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(FairDataset(df_val, MAX_LEN), batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(FairDataset(df_test, MAX_LEN), batch_size=64, shuffle=False, num_workers=0)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class FairRobertaAdv(nn.Module):
    def __init__(self, base_model, num_labels, num_groups, dropout=0.1, hidden_mid=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.task_head = nn.Linear(hidden, num_labels)
        self.adv_head = nn.Linear(hidden, num_groups)
        self.rationale_head = nn.Sequential(
            nn.Linear(hidden, hidden_mid),
            nn.GELU(),
            nn.Linear(hidden_mid, 1)
        )
    def forward(self, input_ids, attention_mask, lambd_adv=1.0):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        cls = self.drop(seq[:, 0])
        task_logits = self.task_head(cls)
        adv_logits = self.adv_head(grad_reverse(cls, lambd_adv))
        rat_logits = self.rationale_head(seq).squeeze(-1)
        return {"task_logits": task_logits, "adv_logits": adv_logits, "rat_logits": rat_logits}

def task_loss_fn(logits, labels):
    return nn.CrossEntropyLoss()(logits, labels)

def adv_loss_fn(logits, groups):
    return nn.CrossEntropyLoss()(logits, groups)

def rationale_loss_fn(rat_logits, rat_mask, attention_mask):
    rat_logits = rat_logits[:, :rat_mask.size(1)]
    attn = attention_mask[:, :rat_mask.size(1)].float()
    bce = nn.BCEWithLogitsLoss(reduction="none")(rat_logits, rat_mask)
    return (bce * attn).sum() / attn.sum().clamp_min(1.0)

def group_rationale_loss_variance(rat_logits, rat_mask, groups, attention_mask):
    rat_logits = rat_logits[:, :rat_mask.size(1)]
    attn = attention_mask[:, :rat_mask.size(1)].float()
    bce = nn.BCEWithLogitsLoss(reduction="none")(rat_logits, rat_mask)
    per_sample = (bce * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1.0)
    group_means = []
    for g in torch.unique(groups):
        idx = groups == g
        if idx.sum() > 1:
            group_means.append(per_sample[idx].mean())
    if len(group_means) < 2:
        return torch.tensor(0.0, device=rat_logits.device)
    return torch.stack(group_means).var(unbiased=False)

def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"], lambd_adv=0.0)
            pred = out["task_logits"].argmax(-1)
            ys.extend(batch["labels"].cpu().tolist())
            ps.extend(pred.cpu().tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")

def train_one_run(lambda_fair=0.0, lambda_adv=0.5, lambda_rat=0.2, fairness_every=10, epochs=3, lr=2e-5):
    model = FairRobertaAdv("roberta-base", 3, len(PRIMARY)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.1 * total_steps)), total_steps)

    best_val_f1 = -1
    best_state = None
    history = {
        "train_loss": [],
        "val_f1": [],
        "val_acc": [],
        "fair_loss": [],
        "rat_loss": [],
        "adv_loss": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = run_fair = run_rat = run_adv = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()

            out = model(batch["input_ids"], batch["attention_mask"], lambd_adv=lambda_adv)
            loss_task = task_loss_fn(out["task_logits"], batch["labels"])
            loss_adv = adv_loss_fn(out["adv_logits"], batch["groups"])
            loss_rat = rationale_loss_fn(out["rat_logits"], batch["rat_mask"], batch["attention_mask"])

            loss_fair = torch.tensor(0.0, device=DEVICE)
            if step % fairness_every == 0:
                loss_fair = group_rationale_loss_variance(
                    out["rat_logits"], batch["rat_mask"], batch["groups"], batch["attention_mask"]
                )

            loss = loss_task + lambda_adv * loss_adv + lambda_rat * loss_rat + lambda_fair * loss_fair
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item()
            run_fair += loss_fair.item()
            run_rat += loss_rat.item()
            run_adv += loss_adv.item()

        val_acc, val_f1 = evaluate(model, val_loader)
        history["train_loss"].append(run_loss / len(train_loader))
        history["fair_loss"].append(run_fair / len(train_loader))
        history["rat_loss"].append(run_rat / len(train_loader))
        history["adv_loss"].append(run_adv / len(train_loader))
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"train={history['train_loss'][-1]:.4f} | "
            f"rat={history['rat_loss'][-1]:.4f} | "
            f"adv={history['adv_loss'][-1]:.4f} | "
            f"fair={history['fair_loss'][-1]:.4f} | "
            f"val_f1={val_f1:.4f} | val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_f1 = evaluate(model, test_loader)
    return model, history, {
        "lambda_fair": lambda_fair,
        "lambda_adv": lambda_adv,
        "lambda_rat": lambda_rat,
        "best_val_f1": best_val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1
    }

def run_lambda_sweep(lambdas=(0.0, 0.1, 0.3, 0.5, 1.0), lambda_adv=0.5, lambda_rat=0.2, fairness_every=10, epochs=3):
    rows = []
    for lam in lambdas:
        print("\n" + "=" * 80)
        print(f"Running lambda_fair={lam}")
        model, hist, metrics = train_one_run(
            lambda_fair=lam,
            lambda_adv=lambda_adv,
            lambda_rat=lambda_rat,
            fairness_every=fairness_every,
            epochs=epochs
        )
        rows.append(metrics)

        os.makedirs("fairref_output_roberta_adv", exist_ok=True)
        torch.save(model.state_dict(), f"fairref_output_roberta_adv/best_model_lambda_{lam}.pt")

    out = pd.DataFrame(rows)
    out.to_csv("fairref_output_roberta_adv/lambda_ablation_results.csv", index=False)
    return out

lambda_results = run_lambda_sweep(
    lambdas=[0.0, 0.1, 0.3, 0.5, 1.0],
    lambda_adv=0.5,
    lambda_rat=0.2,
    fairness_every=10,
    epochs=3
)

lambda_results
