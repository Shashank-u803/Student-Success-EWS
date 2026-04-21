"""
Script: Fairness Audit
Evaluates model performance separately across demographic sub-groups
to detect and document algorithmic bias.
Outputs a fairness report table and saves a bar chart.
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(DATA_DIR, "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def audit():
    # Load data
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test_unscaled.csv"))
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
    X_scaled = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

    with open(os.path.join(DATA_DIR, "xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    results = []

    # ── 1. Overall baseline ─────────────────────────────────────────
    results.append({
        "Sub-group": "ALL STUDENTS (Overall)",
        "n": len(y_test),
        "Recall": recall_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "Avg Risk Score": probs.mean(),
    })

    # ── 2. Gender ───────────────────────────────────────────────────
    if "Gender" in X_test.columns:
        for g_val, g_label in [(0, "Gender: Female (0)"), (1, "Gender: Male (1)")]:
            mask = X_test["Gender"] == g_val
            if mask.sum() >= 10:
                results.append({
                    "Sub-group": g_label,
                    "n": int(mask.sum()),
                    "Recall": recall_score(y_test[mask], preds[mask]),
                    "Precision": precision_score(y_test[mask], preds[mask], zero_division=0),
                    "F1": f1_score(y_test[mask], preds[mask]),
                    "Avg Risk Score": probs[mask].mean(),
                })

    # ── 3. Scholarship vs Non-Scholarship ───────────────────────────
    if "Scholarship holder" in X_test.columns:
        for s_val, s_label in [(1, "Scholarship: Yes"), (0, "Scholarship: No")]:
            mask = X_test["Scholarship holder"] == s_val
            if mask.sum() >= 10:
                results.append({
                    "Sub-group": s_label,
                    "n": int(mask.sum()),
                    "Recall": recall_score(y_test[mask], preds[mask]),
                    "Precision": precision_score(y_test[mask], preds[mask], zero_division=0),
                    "F1": f1_score(y_test[mask], preds[mask]),
                    "Avg Risk Score": probs[mask].mean(),
                })

    # ── 4. Financial Risk ────────────────────────────────────────────
    if "Financial_Risk" in X_test.columns:
        for r_val, r_label in [(1, "Financial Risk: Yes"), (0, "Financial Risk: No")]:
            mask = X_test["Financial_Risk"] == r_val
            if mask.sum() >= 10:
                results.append({
                    "Sub-group": r_label,
                    "n": int(mask.sum()),
                    "Recall": recall_score(y_test[mask], preds[mask]),
                    "Precision": precision_score(y_test[mask], preds[mask], zero_division=0),
                    "F1": f1_score(y_test[mask], preds[mask]),
                    "Avg Risk Score": probs[mask].mean(),
                })

    # ── 5. Age Groups ───────────────────────────────────────────────
    if "Age at enrollment" in X_test.columns:
        age = X_test["Age at enrollment"]
        for label, mask in [
            ("Age: 17-20 (Young)", (age >= 17) & (age <= 20)),
            ("Age: 21-25 (Mid)", (age >= 21) & (age <= 25)),
            ("Age: 26+ (Mature)", age >= 26),
        ]:
            if mask.sum() >= 10:
                results.append({
                    "Sub-group": label,
                    "n": int(mask.sum()),
                    "Recall": recall_score(y_test[mask], preds[mask]),
                    "Precision": precision_score(y_test[mask], preds[mask], zero_division=0),
                    "F1": f1_score(y_test[mask], preds[mask]),
                    "Avg Risk Score": probs[mask].mean(),
                })

    df_results = pd.DataFrame(results)
    for col in ["Recall", "Precision", "F1", "Avg Risk Score"]:
        df_results[col] = df_results[col].round(3)

    print("\n===== FAIRNESS AUDIT REPORT =====\n")
    print(df_results.to_string(index=False))

    # ── Save CSV ─────────────────────────────────────────────────────
    out_csv = os.path.join(DATA_DIR, "fairness_audit.csv")
    df_results.to_csv(out_csv, index=False)

    # ── Plot Recall by sub-group ─────────────────────────────────────
    plt.rcParams.update({'figure.facecolor':'#161B22','axes.facecolor':'#161B22',
                         'axes.edgecolor':'#30363D','text.color':'#C9D1D9',
                         'xtick.color':'#8B949E','ytick.color':'#8B949E'})

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#58A6FF' if 'Overall' in r else '#E63946' if abs(r['Recall'] - df_results.iloc[0]['Recall']) > 0.05 else '#4CAF83'
              for _, r in df_results.iterrows()]

    bars = ax.barh(df_results["Sub-group"], df_results["Recall"],
                   color=colors, edgecolor='#0D1117', height=0.6)
    ax.axvline(df_results.iloc[0]["Recall"], color='white', lw=1.5, ls='--', alpha=0.6,
               label=f'Overall Recall = {df_results.iloc[0]["Recall"]:.3f}')
    ax.set_xlabel("Recall (Dropout Detection Rate)")
    ax.set_title("Fairness Audit — Dropout Recall Across Sub-groups", fontweight='bold', fontsize=13)
    ax.set_xlim(0, 1.1)
    for bar, val in zip(bars, df_results["Recall"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    ax.legend(facecolor='#161B22', labelcolor='white', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out_plot = os.path.join(PLOTS_DIR, "12_fairness_audit.png")
    plt.savefig(out_plot, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nFairness chart saved: {out_plot}")
    print(f"Fairness CSV saved:   {out_csv}")
    print("\nMax Recall gap across sub-groups:",
          round(df_results["Recall"].max() - df_results["Recall"].min(), 3))

if __name__ == "__main__":
    audit()
