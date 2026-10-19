"""
MACS-Net Training Script
========================
Trains a Random Forest ensemble model simulating the Clinical Feature Encoder (CFE)
branch of the MACS-Net architecture, with:
  - SMOTE oversampling to handle class imbalance (focal-loss equivalent)
  - ColumnTransformer preprocessing pipeline (StandardScaler + OneHotEncoder)
  - Cross-validation for hyperparameter selection
  - Saves trained model + preprocessor as .pkl artifacts

Usage:
    python train.py
"""

import os, warnings, json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, recall_score, precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "data", "healthcare-dataset-stroke-data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stroke_model.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "dataset_stats.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature definitions ────────────────────────────────────────────────────
NUMERIC_FEATURES     = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
TARGET               = "stroke"

print("=" * 60)
print("  MACS-Net — Clinical Feature Encoder (CFE) Training")
print("=" * 60)

# ── 1. Load & Clean ────────────────────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Raw rows: {len(df)}")

# Drop the id column
df.drop(columns=["id"], errors="ignore", inplace=True)

# Fill missing BMI with median (N/A values)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi"].fillna(df["bmi"].median(), inplace=True)

# Remove 'Other' gender (only 1 sample — insufficient)
df = df[df["gender"] != "Other"]

# Convert hypertension / heart_disease to int
df["hypertension"]  = df["hypertension"].astype(int)
df["heart_disease"] = df["heart_disease"].astype(int)
df["stroke"]        = df["stroke"].astype(int)

print(f"      Clean rows: {len(df)}")
print(f"      Stroke: {df['stroke'].sum()} | Non-stroke: {(df['stroke']==0).sum()}")
print(f"      Class ratio: 1:{round((df['stroke']==0).sum()/df['stroke'].sum(), 1)}")

# ── 2. Save dataset statistics for API ────────────────────────────────────
print("\n[2/6] Computing dataset statistics...")
stats = {
    "total_patients": int(len(df)),
    "stroke_cases":   int(df["stroke"].sum()),
    "stroke_rate":    round(df["stroke"].mean() * 100, 2),
    "age_mean":       round(df["age"].mean(), 1),
    "age_std":        round(df["age"].std(), 1),
    "glucose_mean":   round(df["avg_glucose_level"].mean(), 1),
    "glucose_std":    round(df["avg_glucose_level"].std(), 1),
    "bmi_mean":       round(df["bmi"].mean(), 1),
    "bmi_std":        round(df["bmi"].std(), 1),
    "gender_dist":    df["gender"].value_counts().to_dict(),
    "smoking_dist":   df["smoking_status"].value_counts().to_dict(),
    "hypertension_stroke_rate": round(
        df[df["hypertension"]==1]["stroke"].mean()*100, 2),
    "heart_disease_stroke_rate": round(
        df[df["heart_disease"]==1]["stroke"].mean()*100, 2),
}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)
print(f"      Stats saved to {STATS_PATH}")

# ── 3. Feature / Target split ─────────────────────────────────────────────
print("\n[3/6] Preparing features...")
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["hypertension", "heart_disease"]]
y = df[TARGET]

# ── 4. Build preprocessing + model pipeline ───────────────────────────────
print("\n[4/6] Building MACS-Net CFE pipeline...")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("bin", "passthrough", ["hypertension", "heart_disease"]),
])

# ── Ensemble: Random Forest + Gradient Boosting (simulates dual-branch fusion)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",  # focal loss equivalent
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

# Soft-voting ensemble → simulates cross-modal attention fusion
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft",
    weights=[0.6, 0.4]
)

# Full pipeline with SMOTE
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(sampling_strategy=0.35, random_state=42, k_neighbors=5)),
    ("classifier", ensemble)
])

# ── 5. Cross-validate ─────────────────────────────────────────────────────
print("\n[5/6] Cross-validating (5-fold stratified)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
cv_rec = cross_val_score(pipeline, X, y, cv=cv, scoring="recall", n_jobs=-1)
print(f"      ROC-AUC : {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")
print(f"      Recall  : {cv_rec.mean():.4f} +/- {cv_rec.std():.4f}")

# ── 6. Final training & evaluation ────────────────────────────────────────
print("\n[6/6] Training final model on full dataset...")
pipeline.fit(X, y)

# Evaluate on training set (for reporting; use held-out in production)
y_prob = pipeline.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.35).astype(int)  # lower threshold = higher sensitivity

print("\n--- Final Training Metrics ------------------------------")
print(classification_report(y, y_pred, target_names=["No Stroke", "Stroke"]))
print(f"ROC-AUC   : {roc_auc_score(y, y_prob):.4f}")
print(f"Recall    : {recall_score(y, y_pred):.4f}")
print(f"Precision : {precision_score(y, y_pred, zero_division=0):.4f}")
print(f"F1        : {f1_score(y, y_pred):.4f}")

cm = confusion_matrix(y, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Save performance to stats
stats["model_performance"] = {
    "cv_roc_auc_mean":   round(float(cv_roc.mean()), 4),
    "cv_roc_auc_std":    round(float(cv_roc.std()), 4),
    "cv_recall_mean":    round(float(cv_rec.mean()), 4),
    "train_roc_auc":     round(float(roc_auc_score(y, y_prob)), 4),
    "train_recall":      round(float(recall_score(y, y_pred)), 4),
    "train_precision":   round(float(precision_score(y, y_pred, zero_division=0)), 4),
    "train_f1":          round(float(f1_score(y, y_pred)), 4),
    "threshold":         0.35,
}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"\n[OK] Model saved -> {MODEL_PATH}")
print(f"[OK] Stats saved -> {STATS_PATH}")
print("\n" + "="*60)
print("  Training complete. Run: uvicorn main:app --reload")
print("="*60)
