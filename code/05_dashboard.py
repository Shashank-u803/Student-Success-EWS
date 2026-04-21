"""
Student Attrition Early Warning System -- Counselor Dashboard
Light Professional Theme (Notion / Linear inspired)
Clean white background, soft shadows, corporate feel.

Run: streamlit run 05_dashboard.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Attrition -- Early Warning System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Light Professional Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

  /* Hide Streamlit chrome */
  header[data-testid="stHeader"] { display: none !important; }
  [data-testid="stHeader"]       { display: none !important; }
  [data-testid="stToolbar"]      { display: none !important; }
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="baseButton-headerNoPadding"],
  [data-testid="stBaseButton-headerNoPadding"],
  button[aria-label*="sidebar"],
  button[title*="sidebar"],
  button[kind="headerNoPadding"] { display: none !important; }
  .main .block-container { padding-top: 1.5rem !important; }

  /* --- Base --- */
  html, body, [class*="css"], [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main {
      background-color: #F9FAFB !important;
      font-family: 'Inter', sans-serif !important;
      color: #111827;
  }

  /* --- Sidebar --- */
  [data-testid="stSidebar"] {
      background-color: #FFFFFF !important;
      border-right: 1px solid #E5E7EB;
  }
  [data-testid="stSidebar"] * {
      color: #374151 !important;
      font-family: 'Inter', sans-serif !important;
  }

  /* --- KPI Cards --- */
  .kpi-card {
      background: #FFFFFF;
      border: 1px solid #E5E7EB;
      border-radius: 12px;
      padding: 22px 24px;
      text-align: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
      transition: box-shadow 0.2s ease;
  }
  .kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
  .kpi-value { font-size: 2.4rem; font-weight: 900; line-height: 1; margin-bottom: 6px; }
  .kpi-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: #9CA3AF; }

  /* --- Section Headers --- */
  .section-header {
      font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px;
      text-transform: uppercase; color: #9CA3AF;
      border-bottom: 1px solid #F3F4F6;
      padding-bottom: 8px; margin-bottom: 16px;
  }

  /* --- Info Block --- */
  .info-block {
      background: #FFFFFF; border: 1px solid #F3F4F6;
      border-radius: 8px; padding: 14px 16px; margin-bottom: 8px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }

  /* --- Alert Blocks --- */
  .alert-high {
      background: #FEF2F2; border-left: 3px solid #EF4444;
      border-radius: 6px; padding: 12px 16px; margin: 6px 0; color: #111827;
  }
  .alert-medium {
      background: #FFFBEB; border-left: 3px solid #F59E0B;
      border-radius: 6px; padding: 12px 16px; margin: 6px 0; color: #111827;
  }
  .alert-low {
      background: #F0FDF4; border-left: 3px solid #22C55E;
      border-radius: 6px; padding: 12px 16px; margin: 6px 0; color: #111827;
  }

  /* --- Streamlit Widget Overrides --- */
  h1 { color: #111827 !important; font-family: 'Inter', sans-serif !important; font-weight: 900 !important; }
  h2, h3 { color: #1F2937 !important; font-family: 'Inter', sans-serif !important; }
  p, li, span { color: #374151 !important; }
  hr { border-color: #E5E7EB !important; }
  .stDataFrame { border: 1px solid #E5E7EB !important; border-radius: 8px !important; }
  [data-testid="stRadio"] label { font-size: 0.88rem !important; color: #374151 !important; }
  .stAlert { border-radius: 8px !important; }
  div[data-testid="stMetricValue"] { color: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants: Light Theme Colors
# ---------------------------------------------------------------------------
COL_HIGH   = '#EF4444'   # Red-500
COL_MEDIUM = '#F59E0B'   # Amber-500
COL_LOW    = '#22C55E'   # Green-500
COL_ACCENT = '#6366F1'   # Indigo-500
COL_BLUE   = '#3B82F6'   # Blue-500

# Matplotlib light theme
plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#E5E7EB',
    'axes.labelcolor':  '#6B7280',
    'xtick.color':      '#6B7280',
    'ytick.color':      '#6B7280',
    'text.color':       '#374151',
    'grid.color':       '#F3F4F6',
    'grid.linestyle':   '-',
    'grid.alpha':       1.0,
    'font.family':      'sans-serif',
})

# ---------------------------------------------------------------------------
# Data & Model Loading
# ---------------------------------------------------------------------------
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(DATA_DIR, "eda_plots")

@st.cache_resource(show_spinner=False)
def load_assets():
    with open(os.path.join(DATA_DIR, "xgb_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data(show_spinner=False)
def load_data():
    X = pd.read_csv(os.path.join(DATA_DIR, "X_test_unscaled.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
    return X, y

def risk_label(p):
    if p >= 0.6: return "HIGH RISK"
    if p >= 0.4: return "MONITOR"
    return "ON TRACK"

def risk_color(p):
    if p >= 0.6: return COL_HIGH
    if p >= 0.4: return COL_MEDIUM
    return COL_LOW

try:
    model, scaler = load_assets()
    X_test, y_test = load_data()
    X_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    probs = model.predict_proba(X_scaled)[:, 1]
    df = X_test.copy()
    df['Risk_Score'] = probs
    df['Risk_Pct']   = (probs * 100).round(1)
    df['Status']     = [risk_label(p) for p in probs]
    df['Actual']     = ['Dropout' if y == 1 else 'Graduate' for y in y_test]
    df['Student_ID'] = [f"STU-{i+1001:04d}" for i in range(len(df))]
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Model not loaded. Run scripts 01 through 03 first. Error: {e}")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Early Warning System")
    st.markdown('<p style="font-size:0.82rem;color:#9CA3AF !important;">Student Attrition -- ML Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "View",
        ["Overview", "Student Profiles", "What-If Analysis", "Model Insights", "EDA Reports"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div class="section-header">System Info</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#6B7280 !important;">Model: XGBoost v2.1<br>Dataset: UCI -- ID 697<br>Students: 4,424 real records<br>Alert Threshold: P(Dropout) >= 0.60</p>', unsafe_allow_html=True)
    if model_ready:
        st.markdown("---")
        st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.8rem;color:#6B7280 !important;">Test set: {len(probs)} students<br>High Risk: {(probs>=0.6).sum()}<br>Monitor: {((probs>=0.4)&(probs<0.6)).sum()}<br>On Track: {(probs<0.4).sum()}</p>', unsafe_allow_html=True)

# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================
if page == "Overview" and model_ready:
    st.markdown("# Student Attrition Early Warning System")
    st.markdown('<p style="color:#6B7280 !important;margin-top:-10px;">Academic Risk Monitoring -- Counselor Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    n_high  = int((probs >= 0.6).sum())
    n_med   = int(((probs >= 0.4) & (probs < 0.6)).sum())
    n_low   = int((probs < 0.4).sum())
    n_total = len(probs)
    avg_risk = probs.mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{COL_HIGH}">{n_high}</div><div class="kpi-label">High Risk</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{COL_MEDIUM}">{n_med}</div><div class="kpi-label">Monitor</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{COL_LOW}">{n_low}</div><div class="kpi-label">On Track</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{COL_ACCENT}">{n_total}</div><div class="kpi-label">Total Students</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{COL_BLUE}">{avg_risk:.1f}%</div><div class="kpi-label">Avg Risk Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        counts, bins, patches = ax.hist(probs, bins=50, alpha=0.9, edgecolor='white', linewidth=0.5)
        for patch, left in zip(patches, bins[:-1]):
            if left < 0.4:   patch.set_facecolor(COL_LOW)
            elif left < 0.6: patch.set_facecolor(COL_MEDIUM)
            else:            patch.set_facecolor(COL_HIGH)
        ax.axvline(0.6, color='#374151', lw=1.5, ls='--', alpha=0.5, label='Alert Threshold (0.60)')
        ax.set_xlabel('Predicted Dropout Probability')
        ax.set_ylabel('Number of Students')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig, width='stretch')
        plt.close()

    with col_chart2:
        st.markdown('<div class="section-header">Status Breakdown</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sizes  = [n_high, n_med, n_low]
        labels = ['High Risk', 'Monitor', 'On Track']
        colors = [COL_HIGH, COL_MEDIUM, COL_LOW]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(edgecolor='white', linewidth=2.5)
        )
        for t in texts:     t.set_color('#6B7280'); t.set_fontsize(9)
        for t in autotexts: t.set_color('white'); t.set_fontweight('bold'); t.set_fontsize(9)
        st.pyplot(fig, width='stretch')
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">High Priority Students -- Immediate Attention Required</div>', unsafe_allow_html=True)

    top_risk = (
        df[df['Status'] == 'HIGH RISK']
        .nlargest(15, 'Risk_Score')
        [['Student_ID', 'Risk_Pct', 'Status',
          'Curricular units 2nd sem (grade)',
          'Curricular units 2nd sem (approved)',
          'Tuition fees up to date',
          'Scholarship holder',
          'Actual']]
        .rename(columns={
            'Risk_Pct': 'Risk Score (%)',
            'Curricular units 2nd sem (grade)': 'Sem 2 Grade',
            'Curricular units 2nd sem (approved)': 'Units Approved',
            'Tuition fees up to date': 'Fees Paid',
            'Scholarship holder': 'Scholar',
        })
    )
    st.dataframe(top_risk, width='stretch', hide_index=True)

# ===========================================================================
# PAGE 2: STUDENT PROFILES
# ===========================================================================
elif page == "Student Profiles" and model_ready:
    st.markdown("# Student Risk Profiles")
    st.markdown('<p style="color:#6B7280 !important;margin-top:-10px;">Detailed individual academic and financial breakdown</p>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns([1, 3])
    with c1:
        filter_status = st.selectbox("Filter by Status", ["ALL", "HIGH RISK", "MONITOR", "ON TRACK"])
        filtered_df = df if filter_status == "ALL" else df[df['Status'] == filter_status]
        selected_id = st.selectbox("Select Student", filtered_df['Student_ID'].tolist())

    row  = df[df['Student_ID'] == selected_id].iloc[0]
    risk = row['Risk_Score']
    color = risk_color(risk)

    with c2:
        ca, cb, cc, cd = st.columns(4)
        ca.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{risk*100:.1f}%</div><div class="kpi-label">Dropout Risk</div></div>', unsafe_allow_html=True)
        cb.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{COL_BLUE}">{row.get("Curricular units 1st sem (grade)", 0):.1f}</div><div class="kpi-label">Sem 1 Grade</div></div>', unsafe_allow_html=True)
        cc.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{COL_BLUE}">{row.get("Curricular units 2nd sem (grade)", 0):.1f}</div><div class="kpi-label">Sem 2 Grade</div></div>', unsafe_allow_html=True)
        traj = row.get('Grade_Trajectory', 0)
        traj_color = COL_LOW if traj >= 0 else COL_HIGH
        cd.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{traj_color}">{traj:+.1f}</div><div class="kpi-label">Grade Trend</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="section-header">Academic Record</div>', unsafe_allow_html=True)
        metrics = {
            "Sem 1 -- Units Enrolled": int(row.get('Curricular units 1st sem (enrolled)', 0)),
            "Sem 1 -- Units Approved": int(row.get('Curricular units 1st sem (approved)', 0)),
            "Sem 1 -- Grade":          f"{row.get('Curricular units 1st sem (grade)', 0):.2f}",
            "Sem 2 -- Units Enrolled": int(row.get('Curricular units 2nd sem (enrolled)', 0)),
            "Sem 2 -- Units Approved": int(row.get('Curricular units 2nd sem (approved)', 0)),
            "Sem 2 -- Grade":          f"{row.get('Curricular units 2nd sem (grade)', 0):.2f}",
            "Completion Rate S1":     f"{row.get('Completion_Rate_S1', 0):.2f}",
            "Struggle Index S2":      f"{row.get('Struggle_Index', 0):.2f}",
        }
        for k, v in metrics.items():
            st.markdown(f'<div class="info-block" style="display:flex;justify-content:space-between;"><span style="color:#6B7280 !important">{k}</span><span style="font-weight:600;color:#111827 !important">{v}</span></div>', unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="section-header">Financial and Support</div>', unsafe_allow_html=True)
        yes_no = lambda v: ("Yes" if v == 1 else "No")
        fin = {
            "Tuition Fees Up To Date": yes_no(row.get('Tuition fees up to date', 1)),
            "Scholarship Holder":      yes_no(row.get('Scholarship holder', 0)),
            "Debtor Status":           yes_no(row.get('Debtor', 0)),
            "Financial Risk Flag":     yes_no(row.get('Financial_Risk', 0)),
            "Support Score":           f"{row.get('Support_Score', 0):.0f}",
            "Age at Enrollment":       int(row.get('Age at enrollment', 0)),
            "Gender (1=Male)":         int(row.get('Gender', 0)),
            "International":           yes_no(row.get('International', 0)),
        }
        for k, v in fin.items():
            val_color = COL_HIGH if str(v) == 'Yes' and k in ['Debtor Status', 'Financial Risk Flag'] else '#111827'
            st.markdown(f'<div class="info-block" style="display:flex;justify-content:space-between;"><span style="color:#6B7280 !important">{k}</span><span style="font-weight:600;color:{val_color} !important">{v}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:16px;">Recommended Actions</div>', unsafe_allow_html=True)
        if risk >= 0.6:
            st.markdown(f'<div class="alert-high"><b style="color:{COL_HIGH}">URGENT:</b> Schedule counseling session within 48 hours.</div>', unsafe_allow_html=True)
            if row.get('Tuition fees up to date', 1) == 0:
                st.markdown(f'<div class="alert-high"><b style="color:{COL_HIGH}">FINANCIAL:</b> Refer to Financial Aid office immediately.</div>', unsafe_allow_html=True)
            if row.get('Curricular units 2nd sem (grade)', 15) < 10:
                st.markdown(f'<div class="alert-high"><b style="color:{COL_HIGH}">ACADEMIC:</b> Assign peer tutoring program.</div>', unsafe_allow_html=True)
        elif risk >= 0.4:
            st.markdown(f'<div class="alert-medium"><b style="color:{COL_MEDIUM}">MONITOR:</b> Send a proactive welfare check-in email.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alert-medium"><b style="color:{COL_MEDIUM}">REVIEW:</b> Schedule optional drop-in session.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-low"><b style="color:{COL_LOW}">ON TRACK:</b> No immediate action required. Continue monitoring.</div>', unsafe_allow_html=True)

# ===========================================================================
# PAGE 3: WHAT-IF ANALYSIS
# ===========================================================================
elif page == "What-If Analysis" and model_ready:
    st.markdown("# Intervention Simulator")
    st.markdown('<p style="color:#6B7280 !important;margin-top:-10px;">Model the real-world impact of counseling interventions on dropout risk</p>', unsafe_allow_html=True)
    st.markdown("---")

    col_sel, col_ctrl, col_res = st.columns([1, 1, 1])

    with col_sel:
        st.markdown('<div class="section-header">Student Selection</div>', unsafe_allow_html=True)
        filter_opt = st.selectbox("Filter", ["HIGH RISK", "MONITOR", "ON TRACK", "ALL"])
        filtered_df2 = df if filter_opt == "ALL" else df[df['Status'] == filter_opt]
        selected = st.selectbox("Student ID", filtered_df2['Student_ID'].tolist())

    row_orig = df[df['Student_ID'] == selected].drop(
        columns=['Risk_Score', 'Risk_Pct', 'Status', 'Actual', 'Student_ID']
    ).iloc[0].copy()
    original_risk = df[df['Student_ID'] == selected]['Risk_Score'].values[0]

    with col_ctrl:
        st.markdown('<div class="section-header">Simulate Intervention</div>', unsafe_allow_html=True)
        new_grade2  = st.slider("Set Semester 2 Grade", 0.0, 20.0, float(row_orig.get('Curricular units 2nd sem (grade)', 12.0)), 0.5)
        new_fees    = st.selectbox("Tuition Fees Paid?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        new_scholar = st.selectbox("Grant Scholarship?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        new_units2  = st.slider("Set Sem 2 Units Approved", 0, 8, int(row_orig.get('Curricular units 2nd sem (approved)', 3)))

    # Apply changes
    mod = row_orig.copy()
    if 'Curricular units 2nd sem (grade)'    in mod: mod['Curricular units 2nd sem (grade)']    = new_grade2
    if 'Tuition fees up to date'             in mod: mod['Tuition fees up to date']             = new_fees
    if 'Scholarship holder'                  in mod: mod['Scholarship holder']                  = new_scholar
    if 'Curricular units 2nd sem (approved)' in mod: mod['Curricular units 2nd sem (approved)'] = new_units2
    g1 = mod.get('Curricular units 1st sem (grade)', 12)
    if 'Grade_Trajectory'  in mod: mod['Grade_Trajectory']  = new_grade2 - g1
    e2 = mod.get('Curricular units 2nd sem (enrolled)', 6)
    if 'Struggle_Index'    in mod: mod['Struggle_Index']    = new_units2 / (e2 + 1)
    if 'Financial_Risk'    in mod: mod['Financial_Risk']    = int(mod.get('Debtor', 0) == 1 or new_fees == 0)
    if 'Support_Score'     in mod: mod['Support_Score']     = new_scholar + (1 - int(mod.get('Debtor', 0)))

    mod_df  = pd.DataFrame([mod])
    mod_s   = pd.DataFrame(scaler.transform(mod_df), columns=mod_df.columns)
    new_risk = model.predict_proba(mod_s)[0][1]
    change   = new_risk - original_risk

    with col_res:
        st.markdown('<div class="section-header">Simulation Result</div>', unsafe_allow_html=True)
        c1r, c2r = st.columns(2)
        c1r.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{risk_color(original_risk)}">{original_risk*100:.1f}%</div><div class="kpi-label">Current Risk</div></div>', unsafe_allow_html=True)
        c2r.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{risk_color(new_risk)}">{new_risk*100:.1f}%</div><div class="kpi-label">Simulated Risk</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        arrow = "DOWN" if change < 0 else "UP"
        chg_color = COL_LOW if change <= 0 else COL_HIGH
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{chg_color}">{change*100:+.1f}%</div><div class="kpi-label">Change ({arrow})</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if change < -0.15:
            st.markdown(f'<div class="alert-low">EFFECTIVE: These interventions would reduce dropout probability by {abs(change)*100:.0f} percentage points. Implement immediately.</div>', unsafe_allow_html=True)
        elif change < 0:
            st.markdown('<div class="alert-medium">MARGINAL: Slight improvement detected. Combine with additional support measures.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-high">INSUFFICIENT: These changes do not reduce risk. Escalate to senior counseling.</div>', unsafe_allow_html=True)

    # Gauge chart
    st.markdown("---")
    st.markdown('<div class="section-header">Risk Gauge Comparison</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
    for i, (label, val) in enumerate([("Current Risk", original_risk), ("Simulated Risk", new_risk)]):
        ax = axes[i]
        color = risk_color(val)
        ax.barh([0], [val], color=color, height=0.5, alpha=0.85)
        ax.barh([0], [1 - val], left=[val], color='#F3F4F6', height=0.5)
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xticks([0, 0.4, 0.6, 1.0])
        ax.set_xticklabels(['0%', '40%', '60%', '100%'])
        ax.set_title(f"{label}: {val*100:.1f}%", color=color, fontweight='bold')
        ax.axvline(0.6, color='#9CA3AF', lw=1, ls='--', alpha=0.6)
        ax.spines[['top', 'right', 'left']].set_visible(False)
    st.pyplot(fig, width='stretch')
    plt.close()

# ===========================================================================
# PAGE 4: MODEL INSIGHTS
# ===========================================================================
elif page == "Model Insights" and model_ready:
    st.markdown("# Model Insights and Evaluation")
    st.markdown('<p style="color:#6B7280 !important;margin-top:-10px;">Performance metrics and explainability from the trained XGBoost model</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
    perf_data = {
        "Metric": ["Dropout Recall", "Dropout Precision", "Dropout F1-Score", "Overall Accuracy", "PR-AUC", "ROC-AUC"],
        "Logistic Regression (Baseline)": ["0.94", "0.88", "0.91", "93%", "0.973", "--"],
        "XGBoost (Primary -- Deployed)":  ["0.93", "0.89", "0.91", "93%", "0.973", "0.975"],
    }
    st.dataframe(pd.DataFrame(perf_data), width='stretch', hide_index=True)
    st.markdown('<p style="font-size:0.85rem;color:#9CA3AF !important;">Trained on 2,904 real student records. Evaluated on 726 held-out test records. Dataset: UCI ML Repository -- ID 697.</p>', unsafe_allow_html=True)

    st.markdown("---")
    plots_to_show = [
        ('08b_confusion_matrix_xgb.png', 'XGBoost Confusion Matrix'),
        ('09_precision_recall_curve.png', 'Precision-Recall Curve'),
        ('10_feature_importance.png', 'Feature Importance (XGBoost)'),
        ('11_shap_summary.png', 'SHAP Feature Impact'),
    ]
    c1, c2 = st.columns(2)
    for i, (fname, caption) in enumerate(plots_to_show):
        fpath = os.path.join(PLOTS_DIR, fname)
        col = c1 if i % 2 == 0 else c2
        if os.path.exists(fpath):
            col.markdown(f'<div class="section-header">{caption}</div>', unsafe_allow_html=True)
            col.image(fpath, width='stretch')

    # Fairness Audit
    fairness_path = os.path.join(PLOTS_DIR, '12_fairness_audit.png')
    if os.path.exists(fairness_path):
        st.markdown("---")
        st.markdown('<div class="section-header">Fairness Audit -- Recall Across Demographic Sub-groups</div>', unsafe_allow_html=True)
        st.image(fairness_path, width='stretch')
        fairness_csv = os.path.join(DATA_DIR, 'fairness_audit.csv')
        if os.path.exists(fairness_csv):
            fa_df = pd.read_csv(fairness_csv)
            st.dataframe(fa_df, width='stretch', hide_index=True)

    # Architecture and Pipeline Diagrams
    st.markdown("---")
    st.markdown('<div class="section-header">System Architecture and Data Pipeline</div>', unsafe_allow_html=True)
    arch_c1, arch_c2 = st.columns(2)
    arch_path = os.path.join(PLOTS_DIR, '13_architecture_diagram.png')
    pipe_path = os.path.join(PLOTS_DIR, '14_data_pipeline_diagram.png')
    if os.path.exists(arch_path):
        arch_c1.markdown('<div class="section-header">Deployment Architecture</div>', unsafe_allow_html=True)
        arch_c1.image(arch_path, width='stretch')
    if os.path.exists(pipe_path):
        arch_c2.markdown('<div class="section-header">Data Pipeline</div>', unsafe_allow_html=True)
        arch_c2.image(pipe_path, width='stretch')

# ===========================================================================
# PAGE 5: EDA REPORTS
# ===========================================================================
elif page == "EDA Reports":
    st.markdown("# Exploratory Data Analysis")
    st.markdown('<p style="color:#6B7280 !important;margin-top:-10px;">Visual analysis of the real UCI student dataset (4,424 records)</p>', unsafe_allow_html=True)
    st.markdown("---")

    eda_only_prefixes = ('01', '02', '03', '04', '05', '06', '07')
    eda_plots = sorted([
        f for f in os.listdir(PLOTS_DIR) if f.endswith('.png') and f[:2] in eda_only_prefixes
    ]) if os.path.exists(PLOTS_DIR) else []

    if not eda_plots:
        st.warning("No EDA plots found. Run 04_eda.py first.")
    else:
        for i in range(0, len(eda_plots), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(eda_plots):
                    fname   = eda_plots[i + j]
                    caption = fname.replace('.png', '').replace('_', ' ').lstrip('0123456789 ').title()
                    fpath   = os.path.join(PLOTS_DIR, fname)
                    cols[j].markdown(f'<div class="section-header">{caption}</div>', unsafe_allow_html=True)
                    cols[j].image(fpath, width='stretch')
