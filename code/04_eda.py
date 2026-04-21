"""
Script 3: EDA — Exploratory Data Analysis
Generates professional-quality graphs saved to data/eda_plots/
These plots are embedded directly into the Technical Report.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(DATA_DIR, "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE   = {'Dropout': '#E63946', 'Graduate': '#2EC4B6', 'Enrolled': '#8338EC'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150})

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "student_attrition_raw.csv"))
    df_binary = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
    df_binary['Label'] = df_binary['Target']

    print("Generating EDA Plots...\n")

    # ── 1. Target Distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df['Target'].value_counts()
    colors = [PALETTE.get(t, '#aaa') for t in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Target Class Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Students')
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[['top','right']].set_visible(False)
    savefig('01_target_distribution.png')

    # ── 2. Grade Distribution (Sem 1 vs Sem 2) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, col in enumerate(['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']):
        for tgt in ['Graduate', 'Dropout']:
            subset = df_binary[df_binary['Label'] == tgt][col]
            axes[i].hist(subset, bins=30, alpha=0.6, color=PALETTE[tgt], label=tgt, edgecolor='white')
        axes[i].set_title(f'{"Semester 1" if i==0 else "Semester 2"} Grade Distribution', fontweight='bold')
        axes[i].set_xlabel('Grade (0–20)')
        axes[i].set_ylabel('Count')
        axes[i].legend()
        axes[i].spines[['top','right']].set_visible(False)
    plt.suptitle('Grade Distributions: Dropouts vs Graduates', fontsize=13, fontweight='bold', y=1.02)
    savefig('02_grade_distributions.png')

    # ── 3. Grade Trajectory Boxplot ────────────────────────────────────
    df_binary['Grade_Trajectory'] = (df_binary['Curricular units 2nd sem (grade)']
                                     - df_binary['Curricular units 1st sem (grade)'])
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df_binary[df_binary['Label']=='Graduate']['Grade_Trajectory'],
            df_binary[df_binary['Label']=='Dropout']['Grade_Trajectory']]
    bp = ax.boxplot(data, labels=['Graduate', 'Dropout'], patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(PALETTE['Graduate'])
    bp['boxes'][1].set_facecolor(PALETTE['Dropout'])
    ax.axhline(0, color='grey', lw=1, ls='--', alpha=0.5)
    ax.set_title('Grade Trajectory (Sem 2 − Sem 1)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Grade Change')
    ax.spines[['top','right']].set_visible(False)
    savefig('03_grade_trajectory.png')

    # ── 4. Correlation Heatmap ─────────────────────────────────────────
    numeric_cols = ['Age at enrollment','Admission grade',
                    'Curricular units 1st sem (approved)','Curricular units 1st sem (grade)',
                    'Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)',
                    'Tuition fees up to date','Scholarship holder','Debtor','GDP']
    existing = [c for c in numeric_cols if c in df_binary.columns]
    df_binary['Target_Num'] = df_binary['Target'].map({'Dropout':1,'Graduate':0})
    corr_df = df_binary[existing + ['Target_Num']].rename(columns={'Target_Num':'⚠️ Dropout Risk'})
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix (incl. Dropout Risk)', fontsize=13, fontweight='bold', pad=15)
    savefig('04_correlation_heatmap.png')

    # ── 5. Financial Risk vs Dropout ───────────────────────────────────
    if 'Debtor' in df.columns and 'Tuition fees up to date' in df.columns:
        df_binary['Financial_Risk'] = ((df_binary['Debtor']==1) | (df_binary['Tuition fees up to date']==0)).astype(int)
        fin_risk = df_binary.groupby(['Financial_Risk', 'Label']).size().unstack(fill_value=0)
        fin_risk.index = ['No Financial Risk', 'Financial Risk']
        fig, ax = plt.subplots(figsize=(7, 4))
        fin_risk[['Dropout','Graduate']].plot(kind='bar', ax=ax, color=[PALETTE['Dropout'], PALETTE['Graduate']],
                                               edgecolor='white', width=0.6)
        ax.set_title('Financial Risk vs Dropout Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Students')
        ax.set_xlabel('')
        ax.set_xticklabels(fin_risk.index, rotation=0)
        ax.spines[['top','right']].set_visible(False)
        savefig('05_financial_risk.png')

    # ── 6. Age at Enrollment Distribution ─────────────────────────────
    if 'Age at enrollment' in df_binary.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        for tgt in ['Graduate', 'Dropout']:
            ages = df_binary[df_binary['Label']==tgt]['Age at enrollment']
            ax.hist(ages, bins=range(17, 60), alpha=0.6, color=PALETTE[tgt], label=tgt, edgecolor='white')
        ax.set_title('Age at Enrollment by Outcome', fontsize=13, fontweight='bold')
        ax.set_xlabel('Age at Enrollment')
        ax.set_ylabel('Count')
        ax.legend()
        ax.spines[['top','right']].set_visible(False)
        savefig('06_age_distribution.png')

    # ── 7. Approval Rate Comparison ────────────────────────────────────
    metrics = {
        'Sem 1 Approval Rate': ('Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)'),
        'Sem 2 Approval Rate': ('Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (enrolled)'),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (title, (num_col, denom_col)) in enumerate(metrics.items()):
        if num_col in df_binary.columns and denom_col in df_binary.columns:
            df_binary[title] = df_binary[num_col] / (df_binary[denom_col] + 1)
        else:
            continue
        for tgt in ['Graduate', 'Dropout']:
            subset = df_binary[df_binary['Label']==tgt][title].dropna()
            axes[i].hist(subset, bins=20, alpha=0.65, color=PALETTE[tgt], label=tgt, edgecolor='white')
        axes[i].set_title(title, fontweight='bold')
        axes[i].set_xlabel('Approval Rate (0–1)')
        axes[i].legend()
        axes[i].spines[['top','right']].set_visible(False)
    plt.suptitle('Course Unit Approval Rates by Semester', fontsize=13, fontweight='bold', y=1.02)
    savefig('07_approval_rates.png')

    print(f"\n✅ All EDA plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
