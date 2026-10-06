"""
AI Powered Malicious Link Defuser
Iteration 1 (baseline) vs Iteration 2 (refined) Random Forest Classifiers
Dataset: Vrbancic et al., "Phishing Websites Dataset" (88,647 URLs, 111 features)
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)

DATA_PATH = "../data/dataset_full.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------
# Iteration 1: purely lexical/structural features computable
# INSTANTLY from the raw URL string alone -- no network calls,
# no WHOIS lookup, no DNS resolution. This is what a real-time
# "link defuser" would check before a user even clicks.
BASELINE_FEATURES = [
    "qty_dot_url", "qty_hyphen_url", "qty_underline_url", "qty_slash_url",
    "qty_questionmark_url", "qty_equal_url", "qty_at_url", "qty_and_url",
    "qty_exclamation_url", "length_url", "qty_tld_url",
    "domain_length", "domain_in_ip", "qty_params", "email_in_url",
    "url_shortened",
]

TARGET = "phishing"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def evaluate(model, X_test, y_test, label):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    cm = confusion_matrix(y_test, preds)
    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")
    print("confusion matrix [[TN FP][FN TP]]:\n", cm)
    return metrics, cm, probs


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc(results, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, (y_test, probs) in results.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Baseline vs Refined")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_feature_importance(model, feature_names, title, path, top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(idx)), importances[idx], color="#1b4f8a")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    df = load_data()
    y = df[TARGET]

    # ============= ITERATION 1: BASELINE =============
    X1 = df[BASELINE_FEATURES]
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    rf1 = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf1.fit(X1_train, y1_train)
    m1, cm1, probs1 = evaluate(rf1, X1_test, y1_test, "Iteration 1: Baseline (16 lexical features)")

    # ============= ITERATION 2: REFINED =============
    X2 = df.drop(columns=[TARGET])
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    rf2 = RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf2.fit(X2_train, y2_train)
    m2, cm2, probs2 = evaluate(rf2, X2_test, y2_test, "Iteration 2: Refined (111 features, tuned)")

    # ---- figures ----
    plot_confusion(cm1, "Confusion Matrix — Iteration 1", "../figures/fig6_1_confusion_baseline.png")
    plot_confusion(cm2, "Confusion Matrix — Iteration 2", "../figures/fig6_2_confusion_refined.png")
    plot_roc(
        {"Iteration 1 (baseline)": (y1_test, probs1), "Iteration 2 (refined)": (y2_test, probs2)},
        "../figures/fig6_3_roc_curve.png",
    )
    plot_feature_importance(rf1, BASELINE_FEATURES, "Top Features — Iteration 1 (Baseline)",
                             "../figures/fig6_4_feature_importance_baseline.png")
    plot_feature_importance(rf2, list(X2.columns), "Top Features — Iteration 2 (Refined)",
                             "../figures/fig6_5_feature_importance_refined.png")

    # ---- save models ----
    joblib.dump(rf1, "../models/rf_baseline.joblib")
    joblib.dump(rf2, "../models/rf_refined.joblib")
    joblib.dump(BASELINE_FEATURES, "../models/baseline_features.joblib")
    joblib.dump(list(X2.columns), "../models/refined_features.joblib")

    # ---- save results table for the report ----
    results = {
        "dataset": {
            "name": "Phishing Websites Dataset (Vrbancic, Fister, Podgorelec, 2020)",
            "total_urls": int(len(df)),
            "legitimate": int((y == 0).sum()),
            "phishing": int((y == 1).sum()),
        },
        "iteration_1_baseline": m1,
        "iteration_2_refined": m2,
    }
    with open("../results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json, models/, and figures/")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
