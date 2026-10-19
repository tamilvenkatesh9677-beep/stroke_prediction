# MACS-Net | Encephalic Stroke Risk Intelligence Platform

MACS-Net (**Multi-modal Attention-driven Clinical and Spatial Network**) is a clinical decision support system that uses machine learning to predict encephalic stroke risk and generate personalized, clinically prioritized prevention roadmaps.

This platform is a production-grade implementation trained on the Kaggle Stroke Dataset (5,110 patient records) utilizing a simulated **Clinical Feature Encoder (CFE)** branch and a **Cross-Modal Attention Fusion** layer represented by a soft-voting ensemble classifier (Random Forest + Gradient Boosting).

---

## Key Features

1. **Dual-Branch CFE Encoder:** Preprocesses and evaluates 11 structured clinical features (Age, Gender, Hypertension, Heart Disease, Marital Status, Work Type, Residence, Blood Glucose, BMI, and Smoking History).
2. **SMOTE Resampling:** Balances the high class imbalance (1:19.5 stroke ratio in raw dataset) to achieve high recall (~94% training set recall, ~85% cross-validation recall) for critical stroke risk identification.
3. **Real-time Inference Preview:** Intercepts form inputs and runs live, debounced inference against the backend API as the user fills out the form.
4. **Interactive SVG Risk Gauge:** Animates dynamically based on predicted risk, categorizing patients into Low, Moderate, High, or Critical risk.
5. **Personalized Prevention & Recovery Roadmap:** Tailors evidence-based clinical actions (DASH/Mediterranean diets, exercise regimens, cardiology follow-ups, and smoking cessation) directly to the patient's risk factor analysis.
6. **Downloadable Clinical Report:** Exports the comprehensive assessment and roadmap as a cleanly formatted, print-ready PDF using `html2pdf.js`.

---

## Folder Structure

```
stroke_prediction/
├── backend/
│   ├── main.py                 # FastAPI application (REST endpoints)
│   ├── train.py                # Model training, preprocessing, & evaluation script
│   ├── requirements.txt        # Backend dependencies
│   ├── data/
│   │   └── healthcare-dataset-stroke-data.csv  # Kaggle Stroke Dataset
│   └── models/
│       ├── stroke_model.pkl    # Serialized trained ensemble model
│       └── dataset_stats.json  # Precomputed dataset statistics and training metrics
├── frontend/
│   ├── index.html              # Core single-page application (SPA) UI
│   ├── style.css               # Modern glassmorphism dark-mode stylesheet
│   └── app.js                  # Frontend client API integration & DOM controller
├── start.bat                   # One-click startup script (Windows)
└── README.md                   # This documentation file
```

---

## Quick Start (One-Click Launch)

To start both the FastAPI backend and the web browser frontend, simply run the `start.bat` file in the root folder:

1. Double-click `start.bat` (or run it via PowerShell/Command Prompt: `.\start.bat`).
2. The script will automatically verify your Python installation, check and install any missing dependencies, start the FastAPI server on `http://localhost:8000`, and launch the web app.
3. To stop the backend server, press any key in the Command Prompt launcher window.

---

## Manual Setup

If you prefer to run the components manually:

### 1. Backend Setup & Training
```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python train.py

# Start the FastAPI server
uvicorn main:app --port 8000
```
The API documentation will be available at `http://localhost:8000/docs`.

### 2. Frontend Launch
Simply open `frontend/index.html` in any modern web browser. It will automatically connect to the backend server running on port `8000`.

---

## Model & Architecture Specs

- **Class Balancer:** SMOTE (Synthetic Minority Over-sampling Technique) target ratio set to 0.35.
- **Ensemble Classifier:** Soft-voting ensemble combining:
  - **Random Forest Classifier** (300 estimators, max depth 12, balanced class weights) - 60% weight
  - **Gradient Boosting Classifier** (200 estimators, learning rate 0.08, max depth 5) - 40% weight
- **Evaluation Metrics (Cross-Validation):**
  - **ROC-AUC:** ~80.5%
  - **Recall (Stroke Class):** ~85.0%
- **Threshold Tuning:** Inference classification uses a decision threshold of `0.35` to prioritize high sensitivity (recall) for clinical safety.
