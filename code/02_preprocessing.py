"""
Script 2: Preprocessing & Feature Engineering
Cleans the full 37-column dataset, engineers temporal/risk features,
and produces train/test splits ready for modeling.
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def preprocess():
    input_file = os.path.join(DATA_DIR, "student_attrition_raw.csv")
    df = pd.read_csv(input_file)

    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nTarget distribution (raw):\n{df['Target'].value_counts()}")

    # ─────────────────────────────────────────────
    # Step 1: Binary target — Dropout vs Graduate only
    # ─────────────────────────────────────────────
    n_before = len(df)
    df = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
    print(f"\nFiltered 'Enrolled' → {n_before - len(df)} rows removed. Remaining: {len(df)}")
    df['Target_Binary'] = df['Target'].map({'Dropout': 1, 'Graduate': 0})

    # ─────────────────────────────────────────────
    # Step 2: Feature Engineering — Temporal Trends
    # ─────────────────────────────────────────────
    df['Grade_Trajectory']    = (df['Curricular units 2nd sem (grade)']
                                 - df['Curricular units 1st sem (grade)'])
    df['Approval_Trajectory'] = (df['Curricular units 2nd sem (approved)']
                                 - df['Curricular units 1st sem (approved)'])
    df['Struggle_Index']      = (df['Curricular units 2nd sem (approved)']
                                 / (df['Curricular units 2nd sem (enrolled)'] + 1))
    df['Completion_Rate_S1']  = (df['Curricular units 1st sem (approved)']
                                 / (df['Curricular units 1st sem (enrolled)'] + 1))
    df['Completion_Rate_S2']  = df['Struggle_Index']  # alias for clarity
    df['Financial_Risk']      = ((df['Debtor'] == 1) | (df['Tuition fees up to date'] == 0)).astype(int)
    df['Support_Score']       = df['Scholarship holder'] + (1 - df['Debtor'])

    print("\nEngineered features added:")
    eng = ['Grade_Trajectory','Approval_Trajectory','Struggle_Index',
           'Completion_Rate_S1','Financial_Risk','Support_Score']
    for f in eng:
        print(f"  • {f}")

    # ─────────────────────────────────────────────
    # Step 3: Drop unused/redundant columns
    # ─────────────────────────────────────────────
    drop_cols = ['Target', 'Target_Binary', 'Completion_Rate_S2']
    X = df.drop(columns=drop_cols)
    y = df['Target_Binary']

    # ─────────────────────────────────────────────
    # Step 4: Stratified 80/20 split (no leakage)
    # ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ─────────────────────────────────────────────
    # Step 5: Scale AFTER split (prevents leakage)
    # ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),       columns=X_test.columns)

    # Save scaler for use in deployment dashboard
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    X_train_s.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test_s.to_csv(os.path.join(DATA_DIR,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(DATA_DIR,   "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR,    "y_test.csv"),  index=False)

    # Save unscaled test for dashboard use
    X_test.to_csv(os.path.join(DATA_DIR, "X_test_unscaled.csv"), index=False)

    print(f"\n✅ Preprocessing complete.")
    print(f"   Train: {X_train_s.shape[0]} rows | Test: {X_test_s.shape[0]} rows")
    print(f"   Features: {X_train_s.shape[1]}")

if __name__ == "__main__":
    preprocess()
