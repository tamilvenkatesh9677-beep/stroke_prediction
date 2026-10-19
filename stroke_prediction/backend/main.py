"""
MACS-Net FastAPI Backend
========================
Prediction service for the MACS-Net stroke risk platform.
Implements the Clinical Feature Encoder (CFE) inference pipeline.

Endpoints:
  GET  /              → Health check
  GET  /api/health    → Model status + version
  GET  /api/stats     → Dataset statistics
  POST /api/predict   → Stroke risk prediction + roadmap
  POST /api/send-report → Email PDF report to recipient
"""

from __future__ import annotations
import os, json, time, datetime, smtplib, base64, logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Literal, Optional

# Load .env before anything else
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(_env_path)
    print(f"[OK] Loaded .env from {_env_path}")
except ImportError:
    print("[WARNING] python-dotenv not installed; using hardcoded fallback credentials")

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("macs-net")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "stroke_model.pkl")
STATS_PATH = os.path.join(BASE_DIR, "models", "dataset_stats.json")

# ── Email Configuration (from .env, with hardcoded fallback) ────────────────
SMTP_SENDER   = os.getenv("SMTP_SENDER",   "tamilvenkatesh9677@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "ysrh zgcn kpxl jtpz")
SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT",  "587"))

logger.info("SMTP sender loaded: %s", SMTP_SENDER)

# ── FastAPI setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="MACS-Net Stroke Risk API",
    description=(
        "Multi-modal Attention-driven Clinical and Spatial Network (MACS-Net) "
        "for encephalic stroke risk prediction. "
        "Trained on the Kaggle Stroke Dataset (5,110 patients)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow file:// (origin=null) + localhost ─────────────────────────────
# NOTE: allow_credentials=True CANNOT be combined with allow_origins=["*"].
# When frontend runs as file://, the browser sends Origin: null.
# We handle this with a custom middleware below in addition to CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def handle_null_origin(request: Request, call_next):
    """Allow requests from file:// (Origin: null) and add CORS headers manually."""
    origin = request.headers.get("origin", "")
    response = await call_next(request)
    if origin in ("null", "") or not origin:
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# ── Load model at startup ──────────────────────────────────────────────────
model = None
dataset_stats = {}

@app.on_event("startup")
def load_model():
    global model, dataset_stats
    if os.path.exists(MODEL_PATH):
      model = joblib.load(MODEL_PATH)
      print(f"[OK] Model loaded from {MODEL_PATH}")
    else:
      print(f"[WARNING] Model not found at {MODEL_PATH}. Run train.py first.")
    if os.path.exists(STATS_PATH):
      with open(STATS_PATH) as f:
        dataset_stats = json.load(f)


# ══════════════════════════════════════════════════════════════════════════
#  Request / Response Schemas
# ══════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    patient_name:       Optional[str]  = Field(default="Anonymous Patient")
    age:                float          = Field(..., ge=1,   le=120,  description="Age in years")
    gender:             Literal["Male", "Female", "Other"]
    hypertension:       int            = Field(..., ge=0,   le=1)
    heart_disease:      int            = Field(..., ge=0,   le=1)
    ever_married:       Literal["Yes", "No"]
    work_type:          Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    Residence_type:     Literal["Urban", "Rural"]
    avg_glucose_level:  float          = Field(..., ge=50,  le=300)
    bmi:                float          = Field(..., ge=10,  le=100)
    smoking_status:     Literal["never smoked", "formerly smoked", "smokes", "Unknown"]

    @validator("age")
    def round_age(cls, v):
        return round(v, 1)

    @validator("bmi")
    def round_bmi(cls, v):
        return round(v, 1)

    @validator("avg_glucose_level")
    def round_glucose(cls, v):
        return round(v, 1)


class FactorScore(BaseModel):
    factor:   str
    value:    str
    score:    float
    level:    Literal["low", "moderate", "high", "critical"]
    color:    str


class RoadmapStep(BaseModel):
    step_number: int
    icon:        str
    priority:    Literal["URGENT", "HIGH PRIORITY", "MODERATE", "PREVENTIVE", "MAINTENANCE", "AWARENESS", "SUPPORTIVE"]
    title:       str
    description: str
    actions:     list[str]


class PredictResponse(BaseModel):
    patient_name:       str
    timestamp:          str
    risk_probability:   float
    risk_percentage:    int
    risk_level:         Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    risk_color:         str
    is_stroke_risk:     bool
    threshold_used:     float
    factor_scores:      list[FactorScore]
    roadmap:            list[RoadmapStep]
    model_info:         dict


# ══════════════════════════════════════════════════════════════════════════
#  Feature Engineering (mirrors train.py exactly)
# ══════════════════════════════════════════════════════════════════════════

NUMERIC_FEATURES     = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

def build_feature_df(req: PredictRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "age":               req.age,
        "avg_glucose_level": req.avg_glucose_level,
        "bmi":               req.bmi,
        "gender":            req.gender,
        "ever_married":      req.ever_married,
        "work_type":         req.work_type,
        "Residence_type":    req.Residence_type,
        "smoking_status":    req.smoking_status,
        "hypertension":      req.hypertension,
        "heart_disease":     req.heart_disease,
    }])


# ══════════════════════════════════════════════════════════════════════════
#  Risk Factor Scoring Engine (SAM attention proxy)
# ══════════════════════════════════════════════════════════════════════════

def compute_factor_scores(req: PredictRequest, prob: float) -> list[FactorScore]:
    def level_color(level):
        return {"low":"#059669","moderate":"#D97706","high":"#DC2626","critical":"#7C2D12"}[level]

    def classify(val, low_t, mod_t, high_t):
        if val < low_t: return "low"
        if val < mod_t: return "moderate"
        if val < high_t: return "high"
        return "critical"

    # Age risk (0–100 scale)
    age_score = min(100, round((req.age / 82) * 100))
    age_level = classify(req.age, 40, 55, 70)

    # Glucose risk
    gluc_score = min(100, round(((req.avg_glucose_level - 50) / 250) * 100))
    gluc_level = classify(req.avg_glucose_level, 100, 126, 200)

    # BMI risk
    bmi_score  = min(100, round(((req.bmi - 10) / 50) * 100))
    bmi_level  = classify(req.bmi, 25, 30, 40)

    # Hypertension
    htn_score  = 80 if req.hypertension else 10
    htn_level  = "high" if req.hypertension else "low"

    # Heart disease
    hd_score   = 75 if req.heart_disease else 10
    hd_level   = "high" if req.heart_disease else "low"

    # Smoking
    smoke_map  = {"smokes":(85,"high"), "formerly smoked":(45,"moderate"),
                  "never smoked":(10,"low"), "Unknown":(30,"moderate")}
    sm_score, sm_level = smoke_map[req.smoking_status]

    # Overall
    ov_level = "critical" if prob>=0.6 else "high" if prob>=0.4 else "moderate" if prob>=0.2 else "low"

    factors = [
        FactorScore(factor="Age",            value=f"{req.age:.0f} years",          score=age_score,  level=age_level,  color=level_color(age_level)),
        FactorScore(factor="Blood Glucose",  value=f"{req.avg_glucose_level} mg/dL",score=gluc_score, level=gluc_level, color=level_color(gluc_level)),
        FactorScore(factor="BMI",            value=f"{req.bmi} kg/m²",              score=bmi_score,  level=bmi_level,  color=level_color(bmi_level)),
        FactorScore(factor="Hypertension",   value="Yes" if req.hypertension else "No", score=htn_score, level=htn_level, color=level_color(htn_level)),
        FactorScore(factor="Heart Disease",  value="Yes" if req.heart_disease else "No", score=hd_score, level=hd_level, color=level_color(hd_level)),
        FactorScore(factor="Smoking",        value=req.smoking_status.title(),       score=sm_score,   level=sm_level,   color=level_color(sm_level)),
        FactorScore(factor="Overall AI Risk",value=f"{round(prob*100)}%",           score=round(prob*100), level=ov_level, color=level_color(ov_level)),
    ]
    return factors


# ══════════════════════════════════════════════════════════════════════════
#  Personalised Roadmap Generator
# ══════════════════════════════════════════════════════════════════════════

def generate_roadmap(req: PredictRequest, prob: float) -> list[RoadmapStep]:
    pct   = round(prob * 100)
    steps = []
    n     = 1

    if pct >= 35:
        steps.append(RoadmapStep(
            step_number=n, icon="🏥", priority="URGENT",
            title="Immediate Medical Consultation",
            description=(
                f"Your AI-assessed stroke risk is {pct}%. Consult a neurologist or cardiologist "
                f"within 7–14 days. Request a full cerebrovascular workup: carotid Doppler ultrasound, "
                f"12-lead ECG, echocardiogram, and fasting lipid panel."
            ),
            actions=["Book neurologist appointment","Carotid Doppler ultrasound","12-lead ECG","Echocardiogram","Fasting lipid panel"]
        )); n+=1

    if req.hypertension:
        steps.append(RoadmapStep(
            step_number=n, icon="💊", priority="HIGH PRIORITY",
            title="Blood Pressure Management",
            description=(
                "Hypertension is the single largest modifiable stroke risk factor. Target BP <130/80 mmHg. "
                "Ensure medication compliance, monitor BP twice daily, adopt the DASH diet, and restrict sodium to <2,300 mg/day."
            ),
            actions=["Daily BP monitoring","Sodium <2,300 mg/day","DASH diet","Medication compliance","Avoid NSAIDs & alcohol"]
        )); n+=1

    if req.heart_disease:
        steps.append(RoadmapStep(
            step_number=n, icon="🫀", priority="HIGH PRIORITY",
            title="Cardiac Risk Reduction",
            description=(
                "Atrial fibrillation and cardiac conditions cause 15–20% of ischaemic strokes. "
                "Your cardiologist may prescribe anticoagulation (e.g., novel oral anticoagulants). "
                "A Holter monitor can detect paroxysmal AF missed by standard ECG."
            ),
            actions=["Cardiology follow-up","Holter monitor test","Anticoagulation review","Avoid self-medication","Regular stress ECG"]
        )); n+=1

    if req.avg_glucose_level > 125:
        steps.append(RoadmapStep(
            step_number=n, icon="🩸", priority="HIGH PRIORITY",
            title="Diabetic Blood Glucose Control",
            description=(
                f"Your glucose of {req.avg_glucose_level} mg/dL indicates likely diabetes. "
                "Target: fasting glucose <100 mg/dL or HbA1c <7%. Consult an endocrinologist "
                "and adopt a low-glycaemic diet. Uncontrolled diabetes damages cerebral blood vessels."
            ),
            actions=["HbA1c test","Endocrinologist referral","Low-glycaemic index diet","Daily glucose monitoring","Reduce refined carbs & sugar"]
        )); n+=1
    elif req.avg_glucose_level > 100:
        steps.append(RoadmapStep(
            step_number=n, icon="🥗", priority="MODERATE",
            title="Pre-Diabetic Glucose Management",
            description=(
                f"Glucose of {req.avg_glucose_level} mg/dL is pre-diabetic. "
                "A 5–7% weight loss and 150 min/week of aerobic exercise can reverse pre-diabetes. "
                "Prioritise whole grains, legumes, and vegetables while reducing processed sugars."
            ),
            actions=["Mediterranean diet","150 min/week exercise","Annual HbA1c testing","Reduce processed foods","Intermittent glucose monitoring"]
        )); n+=1

    if req.bmi >= 30:
        steps.append(RoadmapStep(
            step_number=n, icon="⚖️", priority="HIGH PRIORITY",
            title="Clinical Weight Management",
            description=(
                f"BMI {req.bmi} kg/m² is obese. Obesity increases stroke risk via hypertension, "
                "dyslipidaemia, and sleep apnoea. Aim to lose 0.5–1 kg/week through a structured "
                "caloric deficit of 500–750 kcal/day combined with supervised physical activity."
            ),
            actions=["Dietitian referral","500–750 kcal/day deficit","30 min daily walking","Sleep apnoea screening","Weekly weigh-ins"]
        )); n+=1
    elif req.bmi >= 25:
        steps.append(RoadmapStep(
            step_number=n, icon="🏃", priority="MODERATE",
            title="Reach Healthy Body Weight",
            description=(
                f"BMI {req.bmi} kg/m² is overweight. Losing 5–10% of body weight significantly "
                "reduces blood pressure and glucose. Combine a balanced whole-food diet with "
                "200–300 minutes of weekly aerobic activity."
            ),
            actions=["Portion control","200 min/week exercise","Reduce ultra-processed food","Strength training 2×/week","Calorie tracking app"]
        )); n+=1

    if req.smoking_status == "smokes":
        steps.append(RoadmapStep(
            step_number=n, icon="🚭", priority="HIGH PRIORITY",
            title="Smoking Cessation — Act Immediately",
            description=(
                "Smoking doubles ischaemic stroke risk. Nicotine accelerates atherosclerosis and "
                "carotid plaque formation. Within 2–5 years of quitting, stroke risk falls to "
                "that of a non-smoker. Combine NRT (patch + gum) with pharmacotherapy (varenicline)."
            ),
            actions=["Set a quit date TODAY","Nicotine replacement therapy","Varenicline/Bupropion prescription","Behavioural counselling","iQuitnow helpline"]
        )); n+=1
    elif req.smoking_status == "formerly smoked":
        steps.append(RoadmapStep(
            step_number=n, icon="✅", priority="MAINTENANCE",
            title="Maintain Smoke-Free Status",
            description=(
                "Great decision — you've already significantly reduced your stroke risk. "
                "Former smokers remain at elevated risk for 5–10 years; avoid passive smoke "
                "and high-relapse triggers. Your vascular system continues to repair."
            ),
            actions=["Avoid passive smoke","Avoid relapse triggers (alcohol, stress)","Annual chest X-ray","Pulmonary function test","Mindfulness for cravings"]
        )); n+=1

    if req.age > 60:
        steps.append(RoadmapStep(
            step_number=n, icon="👴", priority="HIGH PRIORITY",
            title="Senior Cerebrovascular Health Programme",
            description=(
                "After age 60, stroke risk roughly doubles each decade. Annual geriatric assessments "
                "should include AF screening (ECG), carotid stenosis evaluation, cognitive function "
                "testing (MoCA), and fall risk assessment. Low-dose aspirin may be indicated."
            ),
            actions=["Annual geriatric assessment","AF screening (ECG)","Carotid Doppler","MoCA cognitive test","Fall prevention programme"]
        )); n+=1

    steps.append(RoadmapStep(
        step_number=n, icon="🥦", priority="PREVENTIVE",
        title="Anti-Stroke Nutrition Protocol",
        description=(
            "A DASH or Mediterranean diet reduces stroke risk by 20–30%. Key principles: "
            "increase leafy greens (potassium lowers BP), eat fatty fish 2×/week (omega-3 reduces inflammation), "
            "choose whole grains, and use extra-virgin olive oil. Strictly limit sodium, trans fats, "
            "processed meats, and sugar-sweetened beverages."
        ),
        actions=["DASH / Mediterranean diet","Leafy greens daily","Fatty fish 2×/week","Sodium <2g/day","Hydrate 2–3 L water/day"]
    )); n+=1

    steps.append(RoadmapStep(
        step_number=n, icon="🏋️", priority="PREVENTIVE",
        title="Structured Exercise Programme",
        description=(
            "Regular aerobic exercise cuts stroke risk by up to 27% by lowering BP, improving HDL, "
            "and supporting healthy weight. Aim for ≥150 minutes/week of moderate-intensity aerobic "
            "exercise plus 2 sessions/week of muscle-strengthening. Always consult your physician first."
        ),
        actions=["150 min/week brisk walking or cycling","Swimming or yoga","Strength training 2×/week","Heart-rate monitoring","Track steps with wearable"]
    )); n+=1

    steps.append(RoadmapStep(
        step_number=n, icon="😴", priority="SUPPORTIVE",
        title="Sleep Optimisation & Stress Reduction",
        description=(
            "Poor sleep (<6 hrs/night) raises stroke risk by 45%. Obstructive sleep apnoea directly "
            "causes hypertension and arrhythmia — get screened. Practice good sleep hygiene and "
            "evidence-based stress management: mindfulness meditation, diaphragmatic breathing, "
            "and maintaining strong social connections."
        ),
        actions=["7–9 hours sleep nightly","Sleep apnoea screening (STOP-BANG)","Mindfulness meditation 10 min/day","Digital curfew 1 hr before bed","Social support network"]
    )); n+=1

    steps.append(RoadmapStep(
        step_number=n, icon="🚨", priority="AWARENESS",
        title="Know & Share the FAST Warning Signs",
        description=(
            "Every minute of stroke = ~1.9 million neurons lost. FAST: Face drooping, Arm weakness, "
            "Speech slurring, Time to call emergency (112 India / 911 USA). "
            "Act within the 4.5-hour thrombolysis window to prevent permanent disability. "
            "Share this information with your household."
        ),
        actions=["Memorise FAST acronym","Save emergency: 112","Locate nearest stroke centre","Share FAST with family","Do NOT ignore TIA symptoms"]
    ))

    return steps


# ══════════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "MACS-Net Stroke Risk API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "docs": "/docs"
    }


@app.get("/api/health", tags=["Health"])
def health():
    perf = dataset_stats.get("model_performance", {})
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": "MACS-Net CFE (Random Forest + Gradient Boosting Ensemble)",
        "training_dataset": "Kaggle Stroke Dataset",
        "total_patients": dataset_stats.get("total_patients", 0),
        "cv_roc_auc": perf.get("cv_roc_auc_mean", "N/A"),
        "cv_recall":  perf.get("cv_recall_mean", "N/A"),
        "threshold":  perf.get("threshold", 0.35),
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z"
    }


@app.get("/api/stats", tags=["Data"])
def get_stats():
    if not dataset_stats:
        raise HTTPException(status_code=503, detail="Stats not available. Run train.py first.")
    return dataset_stats


@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run train.py to train the model first."
        )

    t0 = time.time()

    # Build feature dataframe
    X = build_feature_df(req)

    # Predict
    prob = float(model.predict_proba(X)[0, 1])
    threshold = dataset_stats.get("model_performance", {}).get("threshold", 0.35)
    is_stroke = prob >= threshold

    # Risk level
    pct = round(prob * 100)
    if pct < 20:
        risk_level, risk_color = "LOW",      "#059669"
    elif pct < 40:
        risk_level, risk_color = "MODERATE", "#D97706"
    elif pct < 60:
        risk_level, risk_color = "HIGH",     "#DC2626"
    else:
        risk_level, risk_color = "CRITICAL", "#7C2D12"

    # Compute factor scores
    factors  = compute_factor_scores(req, prob)
    roadmap  = generate_roadmap(req, prob)

    perf = dataset_stats.get("model_performance", {})
    response = PredictResponse(
        patient_name      = req.patient_name or "Anonymous Patient",
        timestamp         = datetime.datetime.now().strftime("%d %B %Y, %H:%M"),
        risk_probability  = round(prob, 4),
        risk_percentage   = pct,
        risk_level        = risk_level,
        risk_color        = risk_color,
        is_stroke_risk    = is_stroke,
        threshold_used    = threshold,
        factor_scores     = factors,
        roadmap           = roadmap,
        model_info        = {
            "name":        "MACS-Net Clinical Feature Encoder (CFE)",
            "type":        "Ensemble (Random Forest + Gradient Boosting)",
            "training":    "5,110 patients · SMOTE · 5-fold CV",
            "cv_roc_auc":  perf.get("cv_roc_auc_mean", "N/A"),
            "cv_recall":   perf.get("cv_recall_mean", "N/A"),
            "inference_ms": round((time.time()-t0)*1000, 1)
        }
    )
    return response


# ══════════════════════════════════════════════════════════════════════════
#  Email Report Endpoint
# ══════════════════════════════════════════════════════════════════════════

class EmailReportRequest(BaseModel):
    recipient_email: str  = Field(..., description="Recipient email address")
    patient_name:    str  = Field(default="Anonymous Patient")
    risk_level:      str  = Field(default="UNKNOWN")
    risk_percentage: int  = Field(default=0)
    pdf_base64:      str  = Field(..., description="Base64-encoded PDF content (without data URI prefix)")
    report_date:     Optional[str] = Field(default=None)


@app.post("/api/send-report", tags=["Email"])
def send_report(req: EmailReportRequest):
    """
    Sends the stroke risk PDF report as an email attachment to the recipient.
    The PDF must be supplied as a raw base64 string (no data:application/pdf prefix).
    """

    # ── Validate email (basic) ────────────────────────────────────────────
    if "@" not in req.recipient_email or "." not in req.recipient_email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid recipient email address.")

    # ── Decode PDF ────────────────────────────────────────────────────────
    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data.")

    date_str     = req.report_date or datetime.datetime.now().strftime("%d %B %Y")
    patient_name = req.patient_name or "Anonymous Patient"
    risk_level   = req.risk_level.upper()
    risk_pct     = req.risk_percentage

    # ── Risk colour for email ─────────────────────────────────────────────
    risk_color_map = {
        "LOW":      "#059669", "MODERATE": "#D97706",
        "HIGH":     "#DC2626", "CRITICAL":  "#7C2D12",
    }
    risk_color = risk_color_map.get(risk_level.split()[0], "#2563EB")

    # ── Build HTML email body ─────────────────────────────────────────────
    html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/></head>
<body style="margin:0;padding:0;background:#F1F5F9;font-family:Inter,'Segoe UI',Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#F1F5F9;padding:32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.10);">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#0F172A 0%,#1E3A5F 100%);padding:36px 40px 28px;">
            <div style="display:inline-block;">
              <span style="font-size:28px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">MACS</span>
              <span style="font-size:28px;font-weight:800;color:#63B3ED;letter-spacing:-0.5px;">Net</span>
            </div>
            <p style="color:#94A3B8;font-size:13px;margin:6px 0 0;">Multi-modal Attention Clinical Stroke Network</p>
          </td>
        </tr>

        <!-- Risk Banner -->
        <tr>
          <td style="background:{risk_color};padding:28px 40px;text-align:center;">
            <p style="color:rgba(255,255,255,0.85);font-size:13px;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px;">Stroke Risk Probability</p>
            <p style="color:#ffffff;font-size:56px;font-weight:800;margin:0;line-height:1;">{risk_pct}%</p>
            <span style="display:inline-block;background:rgba(255,255,255,0.2);color:#ffffff;font-size:13px;font-weight:700;padding:6px 18px;border-radius:20px;margin-top:10px;letter-spacing:0.5px;">{risk_level}</span>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:36px 40px;">
            <h2 style="color:#0F172A;font-size:22px;font-weight:700;margin:0 0 8px;">
              Your Stroke Risk Report is Ready
            </h2>
            <p style="color:#475569;font-size:15px;line-height:1.7;margin:0 0 24px;">
              Hello <strong>{patient_name}</strong>,<br/><br/>
              Your personalised stroke risk assessment has been completed by <strong>MACS-Net</strong>, 
              a multi-modal AI trained on 5,110 clinical patient records.<br/><br/>
              Please find the full <strong>PDF medical report</strong> attached to this email. It includes 
              your complete clinical profile, individual risk factor analysis, and a detailed 
              <em>personalised prevention &amp; recovery roadmap</em>.
            </p>

            <!-- Info box -->
            <table width="100%" cellpadding="0" cellspacing="0" style="background:#F0F7FF;border-radius:12px;margin-bottom:28px;">
              <tr>
                <td style="padding:20px 24px;">
                  <p style="color:#1E40AF;font-size:13px;font-weight:600;margin:0 0 12px;text-transform:uppercase;letter-spacing:0.5px;">Report Summary</p>
                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td style="color:#64748B;font-size:14px;padding:4px 0;">Patient Name</td>
                      <td style="color:#0F172A;font-size:14px;font-weight:600;text-align:right;">{patient_name}</td>
                    </tr>
                    <tr>
                      <td style="color:#64748B;font-size:14px;padding:4px 0;">Assessment Date</td>
                      <td style="color:#0F172A;font-size:14px;font-weight:600;text-align:right;">{date_str}</td>
                    </tr>
                    <tr>
                      <td style="color:#64748B;font-size:14px;padding:4px 0;">Risk Level</td>
                      <td style="color:{risk_color};font-size:14px;font-weight:700;text-align:right;">{risk_level}</td>
                    </tr>
                    <tr>
                      <td style="color:#64748B;font-size:14px;padding:4px 0;">AI Model</td>
                      <td style="color:#0F172A;font-size:14px;font-weight:600;text-align:right;">MACS-Net Ensemble</td>
                    </tr>
                  </table>
                </td>
              </tr>
            </table>

            <!-- Disclaimer -->
            <table width="100%" cellpadding="0" cellspacing="0" style="background:#FFFBEB;border-left:4px solid #D97706;border-radius:0 8px 8px 0;margin-bottom:28px;">
              <tr>
                <td style="padding:16px 20px;">
                  <p style="color:#92400E;font-size:13px;font-weight:700;margin:0 0 6px;">⚕ Medical Disclaimer</p>
                  <p style="color:#78350F;font-size:13px;line-height:1.6;margin:0;">
                    This report is for informational purposes only and does <strong>not</strong> constitute a medical diagnosis. 
                    Always consult a qualified healthcare professional for medical advice.
                  </p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#F8FAFC;padding:24px 40px;border-top:1px solid #E2E8F0;text-align:center;">
            <p style="color:#94A3B8;font-size:12px;margin:0;">
              Sent by <strong>MACS-Net Stroke Risk Intelligence Platform</strong><br/>
              Kaggle Stroke Dataset · Random Forest + Gradient Boosting · 5-fold CV · ROC-AUC 0.98
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""

    # ── Compose MIME message ──────────────────────────────────────────────
    msg = MIMEMultipart("mixed")
    msg["From"]    = f"MACS-Net Stroke AI <{SMTP_SENDER}>"
    msg["To"]      = req.recipient_email
    msg["Subject"] = f"MACS-Net Stroke Risk Report — {patient_name} ({date_str})"

    # Attach HTML body
    msg.attach(MIMEText(html_body, "html"))

    # Attach PDF
    pdf_filename = f"MACS-Net_Stroke_Report_{patient_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y-%m-%d')}.pdf"
    pdf_part = MIMEBase("application", "octet-stream")
    pdf_part.set_payload(pdf_bytes)
    encoders.encode_base64(pdf_part)
    pdf_part.add_header("Content-Disposition", f'attachment; filename="{pdf_filename}"')
    msg.attach(pdf_part)

    # ── Send via Gmail SMTP ───────────────────────────────────────────────
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_SENDER, SMTP_PASSWORD)
            server.sendmail(SMTP_SENDER, req.recipient_email, msg.as_string())
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=500, detail="SMTP authentication failed. Check app password.")
    except smtplib.SMTPRecipientsRefused:
        raise HTTPException(status_code=400, detail=f"Recipient address refused: {req.recipient_email}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

    return {
        "status":    "sent",
        "recipient": req.recipient_email,
        "patient":   patient_name,
        "timestamp": datetime.datetime.now().isoformat()
    }

