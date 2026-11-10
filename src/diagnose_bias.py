"""
AI Powered Malicious Link Defuser - Bias Diagnostics
Analyzes lexical feature bias (specifically qty_slash_url) on Vrbancic et al. dataset.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

# Resolve dataset path regardless of whether run from src/ or project root
DATA_PATHS = [
    os.path.join("..", "data", "dataset_full.csv"),
    os.path.join("data", "dataset_full.csv"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset_full.csv")),
]

DATA_PATH = None
for p in DATA_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break

if not DATA_PATH:
    print("data/dataset_full.csv is missing.")
    print("Download command from README.md:")
    print("cd malicious_link_defuser/data")
    print("curl -L -o dataset_full.csv https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv")
    sys.exit(1)

# Resolve diagnostics.json destination path (save to ../diagnostics.json if in src/, or diagnostics.json)
if os.path.basename(os.getcwd()) == "src":
    OUT_JSON_PATH = os.path.join("..", "diagnostics.json")
else:
    OUT_JSON_PATH = os.path.join("diagnostics.json")

RANDOM_STATE = 42
TARGET = "phishing"

BASELINE_FEATURES = [
    "qty_dot_url", "qty_hyphen_url", "qty_underline_url", "qty_slash_url",
    "qty_questionmark_url", "qty_equal_url", "qty_at_url", "qty_and_url",
    "qty_exclamation_url", "length_url", "qty_tld_url",
    "domain_length", "domain_in_ip", "qty_params", "email_in_url",
    "url_shortened",
]

def get_bucket(val):
    if val == 0:
        return "0"
    elif val == 1:
        return "1"
    elif val == 2:
        return "2"
    elif 3 <= val <= 4:
        return "3-4"
    else:
        return "5+"

BUCKET_ORDER = ["0", "1", "2", "3-4", "5+"]

def main():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]
    X = df[BASELINE_FEATURES]

    # Split: test_size=0.2, stratify=y, random_state=42 (identical to train.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # (a) Legit/phishing counts per qty_slash_url bucket [0, 1, 2, 3-4, 5+]
    buckets_all = df["qty_slash_url"].apply(get_bucket)
    counts_per_bucket = {}
    for b in BUCKET_ORDER:
        b_mask = (buckets_all == b)
        legit = int(((b_mask) & (y == 0)).sum())
        phish = int(((b_mask) & (y == 1)).sum())
        total = int(b_mask.sum())
        counts_per_bucket[b] = {
            "legitimate": legit,
            "phishing": phish,
            "total": total,
            "phishing_ratio": round(phish / total, 4) if total > 0 else 0.0
        }

    # Train full 16-feature baseline RF model
    rf_full = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_full.fit(X_train, y_train)

    # (d) ROC-AUC of the full 16-feature model
    probs_full = rf_full.predict_proba(X_test)[:, 1]
    preds_full = rf_full.predict(X_test)
    roc_auc_full = round(float(roc_auc_score(y_test, probs_full)), 4)

    # (b) Accuracy, false-positive rate and recall inside each bucket on test set
    buckets_test = X_test["qty_slash_url"].apply(get_bucket)
    bucket_metrics = {}
    for b in BUCKET_ORDER:
        b_idx = (buckets_test == b)
        if b_idx.sum() == 0:
            continue
        y_sub = y_test[b_idx]
        preds_sub = preds_full[b_idx]

        acc = round(float(accuracy_score(y_sub, preds_sub)), 4)

        # Recall (TP / (TP + FN))
        phish_mask = (y_sub == 1)
        if phish_mask.sum() > 0:
            rec = round(float(recall_score(y_sub, preds_sub, zero_division=0)), 4)
        else:
            rec = None

        # False Positive Rate (FP / (FP + TN) = FP / Actual Negatives)
        legit_mask = (y_sub == 0)
        if legit_mask.sum() > 0:
            fp = int(((legit_mask) & (preds_sub == 1)).sum())
            tn = int(((legit_mask) & (preds_sub == 0)).sum())
            fpr = round(float(fp / (fp + tn)), 4)
        else:
            fpr = None

        bucket_metrics[b] = {
            "test_samples": int(b_idx.sum()),
            "accuracy": acc,
            "false_positive_rate": fpr,
            "recall": rec
        }

    # (c) ROC-AUC of single-feature models using only qty_slash_url, only length_url, only domain_length
    single_feature_aucs = {}
    for feat in ["qty_slash_url", "length_url", "domain_length"]:
        rf_single = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_single.fit(X_train[[feat]], y_train)
        probs_single = rf_single.predict_proba(X_test[[feat]])[:, 1]
        auc_single = round(float(roc_auc_score(y_test, probs_single)), 4)
        single_feature_aucs[feat] = auc_single

    diagnostics = {
        "dataset_total": int(len(df)),
        "slash_bucket_counts": counts_per_bucket,
        "test_set_bucket_performance": bucket_metrics,
        "single_feature_roc_auc": single_feature_aucs,
        "full_16_feature_roc_auc": roc_auc_full
    }

    # Print results
    print("\n" + "=" * 60)
    print("DIAGNOSTICS REPORT: LEXICAL SLASH BIAS & ROC-AUC ANALYSIS")
    print("=" * 60)
    print("\n(a) Dataset Counts per qty_slash_url bucket:")
    for b, c in counts_per_bucket.items():
        print(f"  Bucket {b:>3}: legit={c['legitimate']:>6}, phish={c['phishing']:>6}, "
              f"total={c['total']:>6}, phish%={c['phishing_ratio']*100:>5.1f}%")

    print("\n(b) Test Set Performance per qty_slash_url bucket:")
    for b, m in bucket_metrics.items():
        rec_str = f"{m['recall']*100:.2f}%" if m['recall'] is not None else "N/A"
        fpr_str = f"{m['false_positive_rate']*100:.2f}%" if m['false_positive_rate'] is not None else "N/A"
        print(f"  Bucket {b:>3} (n={m['test_samples']:>5}): "
              f"Acc={m['accuracy']*100:>5.2f}%, FPR={fpr_str:>7}, Recall={rec_str:>7}")

    print("\n(c) Single-Feature Model ROC-AUC:")
    for feat, auc in single_feature_aucs.items():
        print(f"  Only {feat:<16}: ROC-AUC = {auc:.4f}")

    print(f"\n(d) Full 16-Feature Baseline Model ROC-AUC: {roc_auc_full:.4f}")
    print("=" * 60)

    # Save to diagnostics.json
    out_path = OUT_JSON_PATH
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nSaved diagnostics to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
