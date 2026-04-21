"""
Script 4: Model Training, Evaluation & Explainability
Trains Logistic Regression (baseline) + XGBoost (primary).
Generates: confusion matrix, PR-AUC curve, feature importance, SHAP plot.
Saves trained model for dashboard use.
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_auc_score
)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(DATA_DIR, "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {'Graduate': '#2EC4B6', 'Dropout': '#E63946'}

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {name}")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Graduate', 'Dropout'],
                yticklabels=['Graduate', 'Dropout'],
                linewidths=1, linecolor='white')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    savefig(filename)

def plot_pr_curve(y_true, y_scores_lr, y_scores_xgb):
    prec_lr, rec_lr, _ = precision_recall_curve(y_true, y_scores_lr)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, y_scores_xgb)
    ap_lr  = average_precision_score(y_true, y_scores_lr)
    ap_xgb = average_precision_score(y_true, y_scores_xgb)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_lr,  prec_lr,  color='#457B9D', lw=2, label=f'Logistic Regression (AP={ap_lr:.2f})')
    ax.plot(rec_xgb, prec_xgb, color='#E63946', lw=2, label=f'XGBoost (AP={ap_xgb:.2f})')
    ax.fill_between(rec_xgb, prec_xgb, alpha=0.08, color='#E63946')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Comparison', fontweight='bold', fontsize=13)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.spines[['top','right']].set_visible(False)
    savefig('09_precision_recall_curve.png')
    return ap_lr, ap_xgb

def main():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # ── 1. Baseline: Logistic Regression ──────────────────────────────
    print("=" * 55)
    print("1. Baseline: Logistic Regression")
    print("=" * 55)
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds  = lr.predict(X_test)
    lr_scores = lr.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, lr_preds, target_names=['Graduate', 'Dropout']))
    plot_confusion_matrix(y_test, lr_preds, 'Confusion Matrix — Logistic Regression',
                          '08a_confusion_matrix_lr.png')

    # ── 2. Primary: XGBoost ───────────────────────────────────────────
    print("=" * 55)
    print("2. Primary Model: XGBoost")
    print("=" * 55)
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    xgb = XGBClassifier(
        random_state=42, base_score=0.5,
        scale_pos_weight=neg/pos,
        eval_metric='logloss',
        max_depth=4, learning_rate=0.05,
        n_estimators=200, subsample=0.8,
        colsample_bytree=0.8, use_label_encoder=False
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_preds  = xgb.predict(X_test)
    xgb_scores = xgb.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, xgb_preds, target_names=['Graduate', 'Dropout']))
    plot_confusion_matrix(y_test, xgb_preds, 'Confusion Matrix — XGBoost',
                          '08b_confusion_matrix_xgb.png')

    # ── 3. PR-AUC Curve ──────────────────────────────────────────────
    ap_lr, ap_xgb = plot_pr_curve(y_test, lr_scores, xgb_scores)
    print(f"\nAP Score → LR: {ap_lr:.3f} | XGBoost: {ap_xgb:.3f}")
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, xgb_scores):.3f}")

    # ── 4. Feature Importance ────────────────────────────────────────
    feat_imp = pd.Series(xgb.feature_importances_, index=X_train.columns)
    top15 = feat_imp.nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    top15.plot(kind='barh', ax=ax, color='#E63946', edgecolor='white')
    ax.set_title('Top 15 Feature Importances (XGBoost)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Importance Score')
    ax.spines[['top','right']].set_visible(False)
    savefig('10_feature_importance.png')

    # ── 5. SHAP Explainability ────────────────────────────────────────
    print("\nGenerating SHAP values...")
    explainer = shap.TreeExplainer(xgb)
    shap_vals  = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_vals, X_test, plot_type='dot', show=False, max_display=15)
    plt.title('SHAP Feature Impact on Dropout Prediction', fontweight='bold', pad=15)
    plt.savefig(os.path.join(PLOTS_DIR, '11_shap_summary.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: 11_shap_summary.png")

    # ── 6. Save Model ────────────────────────────────────────────────
    with open(os.path.join(DATA_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    print(f"\n✅ XGBoost model saved for dashboard deployment.")

if __name__ == "__main__":
    main()
