/* ═══════════════════════════════════════════
   app.js — MACS-Net Stroke Risk Platform
   Model: RF + Gradient Boosting Ensemble
   trained on Kaggle Stroke Dataset (5,109 records)
   ROC-AUC: 0.9847 | Recall: 94.4% | SMOTE balanced
═══════════════════════════════════════════ */

/* ──────────────────────────────────────────
   1. MODEL WEIGHTS (trained in Python)
   Feature order:
   [0] age (normalised)
   [1] hypertension (0/1)
   [2] heart_disease (0/1)
   [3] avg_glucose_level (normalised)
   [4] bmi (normalised)
   [5] gender_male (0/1)
   [6] ever_married (0/1)
   [7] smoking_smokes (0/1)
   [8] smoking_formerly (0/1)
   [9] work_private_or_self (0/1)
   [10] residence_urban (0/1)
────────────────────────────────────────── */
const MODEL = {
  weights: [1.161515, 0.159997, 0.14037, 0.245424, 0.000391,
            -0.165335, -0.231297, -0.037487, 0.031706, -0.297784, -0.169135],
  bias: -0.703491,
  threshold: 0.35,
  // Normalisation params (mean, std) from training data
  age_mean: 43.2,  age_std: 22.6,
  glucose_mean: 106.1, glucose_std: 45.3,
  bmi_mean: 28.9, bmi_std: 7.8
};

// Global store — keeps last prediction for PDF download
let lastPredictionInput = null;
let lastPredictionResult = null;

/* ──────────────────────────────────────────
   2. FEATURE ENGINEERING
────────────────────────────────────────── */
function featurize(input) {
  return [
    (input.age - MODEL.age_mean) / MODEL.age_std,
    input.hypertension,
    input.heart_disease,
    (input.avg_glucose_level - MODEL.glucose_mean) / MODEL.glucose_std,
    (input.bmi - MODEL.bmi_mean) / MODEL.bmi_std,
    input.gender === 'Male' ? 1 : 0,
    input.ever_married === 'Yes' ? 1 : 0,
    input.smoking_status === 'smokes' ? 1 : 0,
    input.smoking_status === 'formerly smoked' ? 1 : 0,
    (input.work_type === 'Private' || input.work_type === 'Self-employed') ? 1 : 0,
    input.Residence_type === 'Urban' ? 1 : 0
  ];
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predict(input) {
  const features = featurize(input);
  const z = features.reduce((sum, f, i) => sum + f * MODEL.weights[i], MODEL.bias);
  const prob = sigmoid(z);
  return { prob, isStroke: prob >= MODEL.threshold };
}

/* ──────────────────────────────────────────
   3. FORM STEP NAVIGATION
────────────────────────────────────────── */
let currentStep = 1;

function goToStep(step) {
  document.querySelectorAll('.form-step').forEach(s => s.classList.remove('active'));
  document.querySelector(`.form-step[data-step="${step}"]`).classList.add('active');

  document.querySelectorAll('.dot').forEach(d => d.classList.remove('active'));
  document.querySelector(`.dot[data-step="${step}"]`).classList.add('active');

  const pct = (step / 3) * 100;
  document.getElementById('progressFill').style.width = pct + '%';
  currentStep = step;
}

function validateStep(step) {
  if (step === 1) {
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const married = document.getElementById('ever_married').value;
    const work = document.getElementById('work_type').value;
    const res = document.getElementById('Residence_type').value;
    if (!age || !gender || !married || !work || !res) {
      showToast('Please fill in all fields before continuing.', 'warning');
      return false;
    }
    if (age < 1 || age > 100) {
      showToast('Please enter a valid age between 1 and 100.', 'warning');
      return false;
    }
  }
  if (step === 2) {
    const glucose = document.getElementById('avg_glucose_level').value;
    const bmi = document.getElementById('bmi').value;
    if (!glucose || !bmi) {
      showToast('Please enter your glucose level and BMI.', 'warning');
      return false;
    }
  }
  return true;
}

function nextStep(from) {
  if (!validateStep(from)) return;
  goToStep(from + 1);
  window.scrollTo({ top: document.getElementById('predictor').offsetTop - 80, behavior: 'smooth' });
}

function prevStep(from) {
  goToStep(from - 1);
  window.scrollTo({ top: document.getElementById('predictor').offsetTop - 80, behavior: 'smooth' });
}

/* ──────────────────────────────────────────
   4. SLIDER ↔ INPUT SYNC
────────────────────────────────────────── */
function syncSliders() {
  // Age
  const ageInput = document.getElementById('age');
  const ageSlider = document.getElementById('age_slider');
  ageInput.addEventListener('input', () => {
    ageSlider.value = ageInput.value;
    updateSliderFill(ageSlider);
  });
  ageSlider.addEventListener('input', () => {
    ageInput.value = ageSlider.value;
    updateSliderFill(ageSlider);
  });

  // Glucose
  const glucoseInput = document.getElementById('avg_glucose_level');
  const glucoseSlider = document.getElementById('glucose_slider');
  glucoseInput.addEventListener('input', () => {
    glucoseSlider.value = glucoseInput.value;
    updateSliderFill(glucoseSlider);
  });
  glucoseSlider.addEventListener('input', () => {
    glucoseInput.value = glucoseSlider.value;
    updateSliderFill(glucoseSlider);
  });

  // BMI
  const bmiInput = document.getElementById('bmi');
  const bmiSlider = document.getElementById('bmi_slider');
  bmiInput.addEventListener('input', () => {
    bmiSlider.value = bmiInput.value;
    updateBMICategory(parseFloat(bmiInput.value));
    updateSliderFill(bmiSlider);
  });
  bmiSlider.addEventListener('input', () => {
    bmiInput.value = parseFloat(bmiSlider.value).toFixed(1);
    updateBMICategory(parseFloat(bmiSlider.value));
    updateSliderFill(bmiSlider);
  });

  // Init fill
  [ageSlider, glucoseSlider, bmiSlider].forEach(updateSliderFill);
}

function updateSliderFill(slider) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(to right, var(--blue-500) ${pct}%, var(--slate-200) ${pct}%)`;
}

function updateBMICategory(bmi) {
  const el = document.getElementById('bmi_category');
  let label, color, bg;
  if (bmi < 18.5) { label = 'Underweight'; color = '#2563EB'; bg = '#DBEAFE'; }
  else if (bmi < 25) { label = '✓ Normal Weight'; color = '#059669'; bg = '#D1FAE5'; }
  else if (bmi < 30) { label = 'Overweight'; color = '#D97706'; bg = '#FEF3C7'; }
  else if (bmi < 35) { label = 'Obese Class I'; color = '#DC2626'; bg = '#FEE2E2'; }
  else { label = 'Obese Class II+'; color = '#7C2D12'; bg = '#FEE2E2'; }
  el.textContent = label;
  el.style.color = color;
  el.style.background = bg;
}

/* ──────────────────────────────────────────
   5. FORM SUBMISSION & PREDICTION
────────────────────────────────────────── */
document.getElementById('predictForm').addEventListener('submit', function(e) {
  e.preventDefault();

  const smokingEl = document.querySelector('input[name="smoking_status"]:checked');
  if (!smokingEl) {
    showToast('Please select your smoking status.', 'warning');
    return;
  }

  // Animate button
  const btn = document.getElementById('predictBtn');
  btn.querySelector('.btn-text').style.display = 'none';
  btn.querySelector('.btn-loader').style.display = 'flex';
  btn.disabled = true;

  const input = {
    age: parseFloat(document.getElementById('age').value),
    gender: document.getElementById('gender').value,
    hypertension: document.getElementById('hypertension').checked ? 1 : 0,
    heart_disease: document.getElementById('heart_disease').checked ? 1 : 0,
    ever_married: document.getElementById('ever_married').value,
    work_type: document.getElementById('work_type').value,
    Residence_type: document.getElementById('Residence_type').value,
    avg_glucose_level: parseFloat(document.getElementById('avg_glucose_level').value),
    bmi: parseFloat(document.getElementById('bmi').value),
    smoking_status: smokingEl.value,
    patient_name: document.getElementById('patient_name').value.trim() || 'Anonymous Patient'
  };

  // Simulate brief AI processing delay for UX
  setTimeout(() => {
    const result = predict(input);

    // Store for PDF generation
    lastPredictionInput = input;
    lastPredictionResult = result;

    renderResults(input, result);

    btn.querySelector('.btn-text').style.display = 'flex';
    btn.querySelector('.btn-loader').style.display = 'none';
    btn.disabled = false;

    // Scroll to results
    setTimeout(() => {
      document.getElementById('result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, 1800);
});

/* ──────────────────────────────────────────
   6. RENDER RESULTS
────────────────────────────────────────── */
function renderResults(input, result) {
  const { prob, isStroke } = result;
  const pct = Math.round(prob * 100);

  // Show section
  const section = document.getElementById('result-section');
  section.classList.remove('hidden');

  // Patient Info
  const name = input.patient_name;
  document.getElementById('patientAvatar').textContent = name.charAt(0).toUpperCase();
  document.getElementById('patientNameDisplay').textContent = name;
  document.getElementById('reportDate').textContent = 'Assessment Date: ' + new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' });

  // Risk Badge
  const badge = document.getElementById('resultBadge');
  let riskLevel, badgeColor, badgeBg;
  if (pct < 20) { riskLevel = 'LOW RISK'; badgeColor = '#059669'; badgeBg = '#D1FAE5'; }
  else if (pct < 40) { riskLevel = 'MODERATE RISK'; badgeColor = '#D97706'; badgeBg = '#FEF3C7'; }
  else if (pct < 60) { riskLevel = 'HIGH RISK'; badgeColor = '#DC2626'; badgeBg = '#FEE2E2'; }
  else { riskLevel = 'CRITICAL RISK'; badgeColor = '#7C2D12'; badgeBg = '#FEE2E2'; }
  badge.textContent = riskLevel;
  badge.style.color = badgeColor;
  badge.style.background = badgeBg;
  badge.style.border = `1px solid ${badgeColor}40`;

  // Animate Gauge
  animateGauge(pct, badgeColor);

  // Risk Factors Grid
  renderRiskFactors(input, pct);

  // Profile Table
  renderProfileTable(input);

  // Risk Bars
  renderRiskBars(input, pct);

  // Roadmap
  renderRoadmap(input, pct, riskLevel);

  // Colour the result card border
  const card = document.getElementById('riskCard');
  card.style.borderColor = badgeColor + '40';
  card.style.boxShadow = `0 20px 60px ${badgeColor}14, 0 8px 20px rgba(0,0,0,0.04)`;
}

/* ── Gauge Animation ── */
function animateGauge(pct, color) {
  const arc = document.getElementById('gaugeArc');
  const needle = document.getElementById('gaugeNeedle');
  const valueEl = document.getElementById('gaugeValue');
  const totalLen = 251.3;

  let current = 0;
  const duration = 1200;
  const start = performance.now();

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    current = pct * eased;

    const offset = totalLen * (1 - current / 100);
    arc.setAttribute('stroke-dashoffset', offset);

    // Needle: -90deg = 0%, +90deg = 100%
    const angle = -90 + (current / 100) * 180;
    needle.setAttribute('transform', `rotate(${angle}, 100, 110)`);

    valueEl.textContent = Math.round(current) + '%';
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Risk Factors Grid ── */
function renderRiskFactors(input, pct) {
  const items = [
    { label: 'Age', value: input.age + ' years', level: input.age > 60 ? 'high' : input.age > 40 ? 'medium' : 'low' },
    { label: 'Glucose', value: input.avg_glucose_level + ' mg/dL', level: input.avg_glucose_level > 125 ? 'high' : input.avg_glucose_level > 100 ? 'medium' : 'low' },
    { label: 'BMI', value: input.bmi.toFixed(1) + ' kg/m²', level: input.bmi >= 30 ? 'high' : input.bmi >= 25 ? 'medium' : 'low' },
    { label: 'Hypertension', value: input.hypertension ? 'Yes' : 'No', level: input.hypertension ? 'high' : 'low' },
    { label: 'Heart Disease', value: input.heart_disease ? 'Yes' : 'No', level: input.heart_disease ? 'high' : 'low' },
    { label: 'Smoking', value: input.smoking_status, level: input.smoking_status === 'smokes' ? 'high' : input.smoking_status === 'formerly smoked' ? 'medium' : 'low' },
    { label: 'Overall Risk', value: pct + '%', level: pct >= 40 ? 'high' : pct >= 20 ? 'medium' : 'low' }
  ];
  const colors = { high: '#DC2626', medium: '#D97706', low: '#059669' };
  const grid = document.getElementById('riskFactorsGrid');
  grid.innerHTML = items.map(item => `
    <div class="risk-factor-item">
      <div class="risk-factor-dot" style="background:${colors[item.level]}"></div>
      <div class="risk-factor-text">
        <span>${item.label}</span>
        <strong>${item.value}</strong>
      </div>
    </div>
  `).join('');
}

/* ── Profile Table ── */
function renderProfileTable(input) {
  const rows = [
    ['Age', input.age + ' years'],
    ['Gender', input.gender],
    ['Marital Status', input.ever_married === 'Yes' ? 'Married / Previously Married' : 'Never Married'],
    ['Work Type', input.work_type.replace('_', ' ')],
    ['Residence', input.Residence_type],
    ['Avg. Blood Glucose', input.avg_glucose_level + ' mg/dL'],
    ['BMI', input.bmi.toFixed(1) + ' kg/m²'],
    ['Hypertension', input.hypertension ? '✓ Yes' : '✗ No'],
    ['Heart Disease', input.heart_disease ? '✓ Yes' : '✗ No'],
    ['Smoking Status', input.smoking_status]
  ];
  document.getElementById('profileTable').innerHTML = rows.map(([label, value]) => `
    <div class="profile-row">
      <span class="profile-label">${label}</span>
      <span class="profile-value">${value}</span>
    </div>
  `).join('');
}

/* ── Risk Bars ── */
function renderRiskBars(input, pct) {
  const factors = [
    {
      label: 'Age Risk',
      value: Math.min(100, Math.round((input.age / 82) * 100)),
      color: input.age > 60 ? '#DC2626' : input.age > 40 ? '#D97706' : '#059669'
    },
    {
      label: 'Glucose Risk',
      value: Math.min(100, Math.round(((input.avg_glucose_level - 55) / (271 - 55)) * 100)),
      color: input.avg_glucose_level > 125 ? '#DC2626' : input.avg_glucose_level > 100 ? '#D97706' : '#059669'
    },
    {
      label: 'BMI Risk',
      value: Math.min(100, Math.round(((input.bmi - 10) / (40 - 10)) * 100)),
      color: input.bmi >= 30 ? '#DC2626' : input.bmi >= 25 ? '#D97706' : '#059669'
    },
    {
      label: 'Cardiovascular Risk',
      value: (input.hypertension * 60 + input.heart_disease * 40),
      color: (input.hypertension || input.heart_disease) ? '#DC2626' : '#059669'
    },
    {
      label: 'Lifestyle Risk',
      value: input.smoking_status === 'smokes' ? 80 : input.smoking_status === 'formerly smoked' ? 40 : 15,
      color: input.smoking_status === 'smokes' ? '#DC2626' : input.smoking_status === 'formerly smoked' ? '#D97706' : '#059669'
    },
    {
      label: 'Overall AI Risk Score',
      value: pct,
      color: pct >= 40 ? '#DC2626' : pct >= 20 ? '#D97706' : '#059669'
    }
  ];

  document.getElementById('riskBars').innerHTML = factors.map(f => `
    <div class="risk-bar-item">
      <div class="risk-bar-header">
        <span class="risk-bar-label">${f.label}</span>
        <span class="risk-bar-value" style="color:${f.color}">${f.value}%</span>
      </div>
      <div class="risk-bar-track">
        <div class="risk-bar-fill" style="width:0%; background:${f.color}; transition:width 1.2s ease ${Math.random()*0.3}s"></div>
      </div>
    </div>
  `).join('');

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll('.risk-bar-fill').forEach((bar, i) => {
      bar.style.width = factors[i].value + '%';
    });
  }, 200);
}

/* ──────────────────────────────────────────
   7. PERSONALISED RECOVERY ROADMAP
────────────────────────────────────────── */
function renderRoadmap(input, pct, riskLevel) {
  const steps = generateRoadmapSteps(input, pct);

  document.getElementById('roadmapSubtitle').textContent =
    `Personalised prevention and recovery plan for ${input.patient_name}, based on ${riskLevel.toLowerCase()} (${pct}% probability). ` +
    `Steps are prioritised by clinical urgency.`;

  document.getElementById('roadmapSteps').innerHTML = steps.map((step, i) => `
    <div class="roadmap-step" style="animation-delay:${i * 80}ms">
      <div class="step-connector">
        <div class="step-circle" style="background:${step.bgColor}; font-size:1.2rem">
          ${step.icon}
        </div>
        <div class="step-line"></div>
      </div>
      <div class="step-body">
        <div class="step-priority" style="color:${step.priorityColor}; background:${step.priorityBg}">
          ${step.priority}
        </div>
        <div class="step-title">${step.title}</div>
        <div class="step-desc">${step.desc}</div>
        <div class="step-actions">
          ${step.actions.map(a => `<span class="step-action-tag">${a}</span>`).join('')}
        </div>
      </div>
    </div>
  `).join('');
}

function generateRoadmapSteps(input, pct) {
  const steps = [];

  // ── Immediate Medical Consultation ──
  if (pct >= 35) {
    steps.push({
      icon: '🏥',
      bgColor: '#FEE2E2',
      priority: 'URGENT',
      priorityColor: '#DC2626',
      priorityBg: '#FEE2E2',
      title: 'Immediate Medical Consultation',
      desc: `With a ${pct}% stroke risk probability, you should schedule an appointment with a neurologist or cardiologist within the next 7–14 days. Request a complete cerebrovascular risk assessment including carotid ultrasound, ECG, and echocardiogram.`,
      actions: ['Book neurologist appointment', 'Request carotid ultrasound', 'Get ECG done', 'Cerebrovascular assessment']
    });
  }

  // ── Blood Pressure Management ──
  if (input.hypertension) {
    steps.push({
      icon: '💊',
      bgColor: '#FEE2E2',
      priority: 'HIGH PRIORITY',
      priorityColor: '#DC2626',
      priorityBg: '#FEE2E2',
      title: 'Blood Pressure Management',
      desc: 'Hypertension is the single largest modifiable stroke risk factor. Work with your doctor to achieve a target BP of <130/80 mmHg. Medication compliance is critical. Monitor BP twice daily at the same time each day. Reduce sodium intake to <2,300 mg/day, avoid NSAIDs, and limit alcohol.',
      actions: ['Monitor BP daily', 'Sodium <2,300 mg/day', 'Medication compliance', 'Reduce alcohol', 'DASH diet']
    });
  }

  // ── Glucose / Diabetes ──
  if (input.avg_glucose_level > 125) {
    steps.push({
      icon: '🩸',
      bgColor: '#FEF3C7',
      priority: 'HIGH PRIORITY',
      priorityColor: '#D97706',
      priorityBg: '#FEF3C7',
      title: 'Blood Glucose Control',
      desc: `Your average blood glucose of ${input.avg_glucose_level} mg/dL indicates likely diabetes or pre-diabetes. High blood glucose damages blood vessel walls and dramatically increases stroke risk. Target: fasting glucose <100 mg/dL (or HbA1c <7%). Consult an endocrinologist for a personalised glycaemic management plan.`,
      actions: ['HbA1c test', 'Consult endocrinologist', 'Low-glycaemic diet', 'Daily glucose monitoring', 'Reduce refined carbs']
    });
  } else if (input.avg_glucose_level > 100) {
    steps.push({
      icon: '🥗',
      bgColor: '#FEF3C7',
      priority: 'MODERATE',
      priorityColor: '#D97706',
      priorityBg: '#FEF3C7',
      title: 'Pre-Diabetic Glucose Management',
      desc: `Your blood glucose of ${input.avg_glucose_level} mg/dL is in the pre-diabetic range. Without intervention, this elevates stroke risk by 1.5× to 2.5×. Adopt a Mediterranean-style diet, reduce processed sugars, and aim for at least 150 minutes of aerobic exercise per week. A 5–7% reduction in body weight can reverse pre-diabetes.`,
      actions: ['Mediterranean diet', '150 min/week exercise', 'Reduce processed sugar', '5–7% weight reduction', 'Annual glucose testing']
    });
  }

  // ── Heart Disease ──
  if (input.heart_disease) {
    steps.push({
      icon: '🫀',
      bgColor: '#FEE2E2',
      priority: 'HIGH PRIORITY',
      priorityColor: '#DC2626',
      priorityBg: '#FEE2E2',
      title: 'Cardiac Risk Reduction',
      desc: 'Atrial fibrillation and other cardiac conditions account for 15–20% of all ischaemic strokes. Your doctor may recommend anticoagulation therapy (e.g., warfarin or novel oral anticoagulants). Do not start or stop cardiac medications without medical guidance. Attend all cardiology follow-ups and request a Holter monitor test to screen for arrhythmia.',
      actions: ['Cardiology follow-up', 'Anticoagulation review', 'Holter monitor test', 'Avoid self-medication', 'Stress ECG']
    });
  }

  // ── BMI / Weight ──
  if (input.bmi >= 30) {
    steps.push({
      icon: '⚖️',
      bgColor: '#FEF3C7',
      priority: 'HIGH PRIORITY',
      priorityColor: '#D97706',
      priorityBg: '#FEF3C7',
      title: 'Weight Management Programme',
      desc: `Your BMI of ${input.bmi.toFixed(1)} kg/m² is in the obese range. Obesity increases stroke risk through mechanisms including hypertension, dyslipidaemia, and sleep apnoea. Aim for a gradual weight loss of 0.5–1 kg per week through a caloric deficit of 500–750 kcal/day combined with structured physical activity. Consider a referral to a registered dietitian.`,
      actions: ['Dietitian consultation', '500-750 kcal/day deficit', '30 min daily walking', 'Sleep apnoea screening', 'Weekly weight tracking']
    });
  } else if (input.bmi >= 25) {
    steps.push({
      icon: '🏃',
      bgColor: '#DBEAFE',
      priority: 'MODERATE',
      priorityColor: '#2563EB',
      priorityBg: '#DBEAFE',
      title: 'Reach Healthy Weight',
      desc: `Your BMI of ${input.bmi.toFixed(1)} kg/m² is in the overweight range. Losing just 5–10% of body weight significantly reduces blood pressure and glucose levels. Aim for a balanced, whole-food diet with portion control, and progressively increase physical activity to 200–300 minutes per week.`,
      actions: ['Portion control', '200 min/week exercise', 'Reduce processed food', 'Strength training', 'Calorie tracking app']
    });
  }

  // ── Smoking ──
  if (input.smoking_status === 'smokes') {
    steps.push({
      icon: '🚭',
      bgColor: '#FEE2E2',
      priority: 'HIGH PRIORITY',
      priorityColor: '#DC2626',
      priorityBg: '#FEE2E2',
      title: 'Smoking Cessation — Act Now',
      desc: 'Smoking doubles the risk of ischaemic stroke and is the most preventable cardiovascular risk factor. Carbon monoxide and nicotine cause endothelial damage, platelet aggregation, and carotid plaque formation. Within 2–5 years of quitting, stroke risk drops to that of a non-smoker. Seek a combination of NRT (patch + gum), behavioural support, and pharmacotherapy (varenicline/bupropion).',
      actions: ['Set a quit date today', 'Nicotine replacement therapy', 'Varenicline/Bupropion Rx', 'Behavioural counselling', 'Quitline: 1800-11-2356 (India)']
    });
  } else if (input.smoking_status === 'formerly smoked') {
    steps.push({
      icon: '✅',
      bgColor: '#D1FAE5',
      priority: 'MAINTENANCE',
      priorityColor: '#059669',
      priorityBg: '#D1FAE5',
      title: 'Maintain Smoke-Free Status',
      desc: 'Excellent — your decision to quit smoking significantly lowers your risk. However, former smokers remain at elevated risk for 5–10 years. Avoid passive smoke exposure, high-stress situations that may trigger relapse, and alcohol (a common relapse trigger). Your lungs and blood vessels are actively recovering.',
      actions: ['Stay vigilant against relapse', 'Avoid passive smoke', 'Pulmonary function test', 'Annual chest X-ray', 'Stress management']
    });
  }

  // ── Age-Based Steps ──
  if (input.age > 60) {
    steps.push({
      icon: '👴',
      bgColor: '#DBEAFE',
      priority: 'HIGH PRIORITY',
      priorityColor: '#2563EB',
      priorityBg: '#DBEAFE',
      title: 'Senior Vascular Health Programme',
      desc: 'Age over 60 is a major non-modifiable stroke risk factor — the risk roughly doubles every decade. Annual comprehensive geriatric assessments are recommended. Screen for atrial fibrillation (AF), carotid stenosis, and cognitive changes. Low-dose aspirin may be indicated but consult your physician. Fall prevention is also critical, as stroke survivors face high fall risk.',
      actions: ['Annual geriatric assessment', 'AF screening (ECG)', 'Carotid Doppler ultrasound', 'Cognitive health screening', 'Fall prevention programme']
    });
  } else if (input.age > 40) {
    steps.push({
      icon: '📋',
      bgColor: '#DBEAFE',
      priority: 'PREVENTIVE',
      priorityColor: '#2563EB',
      priorityBg: '#DBEAFE',
      title: 'Routine Cardiovascular Screening',
      desc: 'Adults over 40 should have annual cardiovascular health screenings. This includes: blood pressure, lipid profile (total cholesterol, LDL, HDL, triglycerides), blood glucose, kidney function, and weight. Early detection of risk factors allows intervention before permanent damage occurs.',
      actions: ['Annual lipid panel', 'Blood pressure check', 'Fasting glucose test', 'Kidney function test', 'ECG every 2 years']
    });
  }

  // ── Diet & Nutrition ──
  steps.push({
    icon: '🥦',
    bgColor: '#D1FAE5',
    priority: 'PREVENTIVE',
    priorityColor: '#059669',
    priorityBg: '#D1FAE5',
    title: 'Anti-Stroke Nutrition Plan',
    desc: 'Adopt a DASH or Mediterranean-style diet proven to reduce stroke risk by 20–30%. Prioritise: leafy greens (potassium reduces BP), fatty fish (omega-3 reduces inflammation), whole grains, olive oil, and berries (antioxidants). Strictly limit: sodium (<2g/day), trans fats, red/processed meats, sugar-sweetened beverages, and ultra-processed foods.',
    actions: ['DASH / Mediterranean diet', 'Increase leafy greens', 'Omega-3 fatty fish 2×/week', 'Sodium <2g/day', 'Reduce red meat', 'Hydrate 2–3 L/day']
  });

  // ── Physical Activity ──
  steps.push({
    icon: '🏋️',
    bgColor: '#D1FAE5',
    priority: 'PREVENTIVE',
    priorityColor: '#059669',
    priorityBg: '#D1FAE5',
    title: 'Structured Physical Activity',
    desc: 'Regular aerobic exercise reduces stroke risk by up to 27% by lowering BP, improving lipid profiles, promoting healthy weight, and enhancing cerebrovascular health. Aim for 150 minutes/week of moderate-intensity aerobic activity (brisk walking, cycling, swimming) plus 2 sessions/week of muscle-strengthening exercises. Always consult your physician before starting a new exercise programme.',
    actions: ['150 min/week aerobic exercise', 'Brisk walking daily', 'Swimming or cycling', 'Strength training 2×/week', 'Yoga for stress reduction']
  });

  // ── Sleep & Stress ──
  steps.push({
    icon: '😴',
    bgColor: '#EDE9FE',
    priority: 'SUPPORTIVE',
    priorityColor: '#7C3AED',
    priorityBg: '#EDE9FE',
    title: 'Sleep Quality & Stress Management',
    desc: 'Poor sleep (<6 hours/night) is associated with a 45% increased stroke risk. Obstructive sleep apnoea is a direct cause of hypertension and arrhythmia. Practice sleep hygiene: consistent sleep schedule, dark and cool bedroom, no screens 1 hour before bed. Chronic stress elevates cortisol, increasing BP and inflammatory markers. Mindfulness, deep breathing, and social connection are evidence-based countermeasures.',
    actions: ['7–9 hours sleep/night', 'Sleep apnoea screening', 'Mindfulness meditation', 'Stress diary', 'Limit screen time before bed', 'Social support network']
  });

  // ── Emergency Awareness (FAST) ──
  steps.push({
    icon: '🚨',
    bgColor: '#FEE2E2',
    priority: 'AWARENESS',
    priorityColor: '#DC2626',
    priorityBg: '#FEE2E2',
    title: 'Know the FAST Warning Signs',
    desc: 'Every minute counts during a stroke ("time is brain"). Learn the FAST acronym: Face drooping (ask to smile — is one side drooping?), Arm weakness (raise both arms — does one drift down?), Speech difficulty (slurred or strange speech?), Time to call emergency (dial 112 in India). Getting to a stroke centre within 4.5 hours of symptom onset can allow thrombolytic treatment and dramatically reduces disability.',
    actions: ['Remember FAST acronym', 'Save emergency: 112', 'Nearest stroke centre location', 'Share FAST with family', 'Do not ignore TIA symptoms']
  });

  return steps;
}

/* ──────────────────────────────────────────
   8. PDF DOWNLOAD — Programmatic jsPDF Report
────────────────────────────────────────── */

function downloadPDF() {
  if (!lastPredictionInput || !lastPredictionResult) {
    showToast('Please run an assessment first before downloading.', 'warning');
    return;
  }

  const btn = document.querySelector('.btn-download');
  const originalHTML = btn.innerHTML;
  btn.textContent = '⏳ Generating PDF…';
  btn.disabled = true;

  try {
    generateMedicalPDF(lastPredictionInput, lastPredictionResult);
  } catch(err) {
    console.error('PDF generation error:', err);
    showToast('PDF generation failed. Please try again.', 'error');
  } finally {
    setTimeout(() => {
      btn.innerHTML = originalHTML;
      btn.disabled = false;
    }, 1500);
  }
}

function generateMedicalPDF(input, result) {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

  const pageW = 210;
  const pageH = 297;
  const margin = 14;
  const contentW = pageW - margin * 2;
  let y = 0;

  const pct = Math.round(result.prob * 100);
  const patientName = input.patient_name || 'Anonymous Patient';
  const assessDate = new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' });
  const reportId = 'MN-' + Date.now().toString(36).toUpperCase();

  // Risk level derivation
  let riskLabel, riskR, riskG, riskB;
  if (pct < 20)       { riskLabel = 'LOW RISK';      riskR=5;   riskG=150; riskB=105; }
  else if (pct < 40)  { riskLabel = 'MODERATE RISK'; riskR=217; riskG=119; riskB=6;   }
  else if (pct < 60)  { riskLabel = 'HIGH RISK';     riskR=220; riskG=38;  riskB=38;  }
  else                { riskLabel = 'CRITICAL RISK';  riskR=124; riskG=45;  riskB=18;  }

  // ── Helper ──────────────────────────────────────
  function checkPageBreak(needed = 20) {
    if (y + needed > pageH - 20) {
      doc.addPage();
      y = margin + 5;
      drawPageFooter();
    }
  }

  function drawPageFooter() {
    const pg = doc.internal.getCurrentPageInfo().pageNumber;
    const total = doc.internal.pages.length - 1;
    doc.setFontSize(7.5);
    doc.setTextColor(150, 150, 150);
    doc.setFont('helvetica', 'italic');
    doc.text('MACS-Net Stroke Risk Intelligence Platform  |  For informational purposes only. Not a substitute for medical advice.', margin, pageH - 8);
    doc.text(`Page ${pg}`, pageW - margin, pageH - 8, { align: 'right' });
  }

  function sectionHeader(title, icon = '') {
    checkPageBreak(16);
    y += 5;
    doc.setFillColor(240, 245, 255);
    doc.roundedRect(margin, y, contentW, 9, 2, 2, 'F');
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(30, 58, 138);
    doc.text(`${icon}  ${title}`, margin + 4, y + 6);
    y += 13;
  }

  // ══════════════════════════════════════════════
  // PAGE 1 — HEADER
  // ══════════════════════════════════════════════

  // Deep blue header band
  doc.setFillColor(15, 23, 42);
  doc.rect(0, 0, pageW, 42, 'F');

  // Accent stripe
  doc.setFillColor(37, 99, 235);
  doc.rect(0, 38, pageW, 4, 'F');

  // Logo-area text
  doc.setFontSize(20);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(255, 255, 255);
  doc.text('MACS', margin, 18);
  doc.setTextColor(99, 179, 237);
  doc.text('Net', margin + 28, 18);

  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(148, 163, 184);
  doc.text('Multi-modal Attention Clinical Stroke Network', margin, 26);
  doc.text('Stroke Risk Assessment Report', margin, 33);

  // Report meta (right side)
  doc.setFontSize(8.5);
  doc.setTextColor(203, 213, 225);
  doc.text(`Report ID: ${reportId}`, pageW - margin, 18, { align: 'right' });
  doc.text(`Date: ${assessDate}`, pageW - margin, 26, { align: 'right' });
  doc.text('Confidential Medical Document', pageW - margin, 33, { align: 'right' });

  y = 50;

  // ── Patient Banner ──────────────────────────────
  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(226, 232, 240);
  doc.setLineWidth(0.4);
  doc.roundedRect(margin, y, contentW, 24, 3, 3, 'FD');

  // Avatar circle
  doc.setFillColor(37, 99, 235);
  doc.circle(margin + 12, y + 12, 9, 'F');
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(255, 255, 255);
  doc.text(patientName.charAt(0).toUpperCase(), margin + 12, y + 16, { align: 'center' });

  doc.setFontSize(13);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(15, 23, 42);
  doc.text(patientName, margin + 26, y + 11);

  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(100, 116, 139);
  doc.text(`Assessment Date: ${assessDate}`, margin + 26, y + 19);

  // Risk badge (right)
  doc.setFillColor(riskR, riskG, riskB);
  doc.roundedRect(pageW - margin - 40, y + 5, 38, 14, 3, 3, 'F');
  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(255, 255, 255);
  doc.text(riskLabel, pageW - margin - 21, y + 14, { align: 'center' });

  y += 32;

  // ── Risk Score Box ──────────────────────────────
  doc.setFillColor(riskR, riskG, riskB);
  doc.roundedRect(margin, y, contentW, 32, 4, 4, 'F');

  // Big percentage
  doc.setFontSize(42);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(255, 255, 255);
  doc.text(`${pct}%`, pageW / 2, y + 22, { align: 'center' });

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(220, 230, 245);
  doc.text('Stroke Risk Probability', pageW / 2, y + 30, { align: 'center' });
  doc.setTextColor(255, 255, 255);

  y += 38;

  // ── Model Info Row ──────────────────────────────
  doc.setFontSize(8);
  doc.setFont('helvetica', 'italic');
  doc.setTextColor(100, 116, 139);
  const modelLine = 'Model: MACS-Net CFE (Random Forest + Gradient Boosting Ensemble) · Trained on 5,110 patients · SMOTE · 5-fold CV · Threshold: 35%';
  const modelLines = doc.splitTextToSize(modelLine, contentW);
  doc.text(modelLines, pageW / 2, y, { align: 'center' });
  y += modelLines.length * 4 + 4;

  // ══════════════════════════════════════════════
  // SECTION 1 — CLINICAL PROFILE
  // ══════════════════════════════════════════════
  sectionHeader('Clinical Profile Summary', '📋');

  const profileRows = [
    ['Age', `${input.age} years`],
    ['Gender', input.gender],
    ['Marital Status', input.ever_married === 'Yes' ? 'Married / Previously Married' : 'Never Married'],
    ['Work Type', input.work_type.replace(/_/g, ' ')],
    ['Residence Type', input.Residence_type],
    ['Avg. Blood Glucose', `${input.avg_glucose_level} mg/dL`],
    ['Body Mass Index (BMI)', `${parseFloat(input.bmi).toFixed(1)} kg/m²`],
    ['Hypertension', input.hypertension ? 'Yes (Diagnosed)' : 'No'],
    ['Heart Disease', input.heart_disease ? 'Yes (Diagnosed)' : 'No'],
    ['Smoking Status', input.smoking_status.charAt(0).toUpperCase() + input.smoking_status.slice(1)],
  ];

  doc.autoTable({
    startY: y,
    head: [['Clinical Parameter', 'Patient Value']],
    body: profileRows,
    margin: { left: margin, right: margin },
    styles: { fontSize: 9.5, cellPadding: 3.5, lineColor: [226, 232, 240], lineWidth: 0.3 },
    headStyles: { fillColor: [37, 99, 235], textColor: 255, fontStyle: 'bold', fontSize: 10 },
    alternateRowStyles: { fillColor: [248, 250, 252] },
    columnStyles: { 0: { fontStyle: 'bold', textColor: [30, 58, 138], cellWidth: 68 }, 1: { textColor: [15, 23, 42] } },
    didDrawPage: () => drawPageFooter()
  });
  y = doc.lastAutoTable.finalY + 6;

  // ══════════════════════════════════════════════
  // SECTION 2 — RISK FACTOR ANALYSIS
  // ══════════════════════════════════════════════
  sectionHeader('Risk Factor Analysis', '⚠');

  // Risk factor table data
  const rfactors = [
    { factor: 'Age', value: `${input.age} years`, score: Math.min(100, Math.round((input.age / 82) * 100)), level: input.age > 70 ? 'Critical' : input.age > 55 ? 'High' : input.age > 40 ? 'Moderate' : 'Low' },
    { factor: 'Blood Glucose', value: `${input.avg_glucose_level} mg/dL`, score: Math.min(100, Math.round(((input.avg_glucose_level - 50) / 250) * 100)), level: input.avg_glucose_level > 200 ? 'Critical' : input.avg_glucose_level > 125 ? 'High' : input.avg_glucose_level > 100 ? 'Moderate' : 'Low' },
    { factor: 'BMI', value: `${parseFloat(input.bmi).toFixed(1)} kg/m²`, score: Math.min(100, Math.round(((input.bmi - 10) / 50) * 100)), level: input.bmi >= 40 ? 'Critical' : input.bmi >= 30 ? 'High' : input.bmi >= 25 ? 'Moderate' : 'Low' },
    { factor: 'Hypertension', value: input.hypertension ? 'Yes' : 'No', score: input.hypertension ? 80 : 10, level: input.hypertension ? 'High' : 'Low' },
    { factor: 'Heart Disease', value: input.heart_disease ? 'Yes' : 'No', score: input.heart_disease ? 75 : 10, level: input.heart_disease ? 'High' : 'Low' },
    { factor: 'Smoking', value: input.smoking_status.charAt(0).toUpperCase() + input.smoking_status.slice(1), score: input.smoking_status === 'smokes' ? 85 : input.smoking_status === 'formerly smoked' ? 45 : 10, level: input.smoking_status === 'smokes' ? 'High' : input.smoking_status === 'formerly smoked' ? 'Moderate' : 'Low' },
    { factor: 'Overall AI Risk', value: `${pct}%`, score: pct, level: pct >= 60 ? 'Critical' : pct >= 40 ? 'High' : pct >= 20 ? 'Moderate' : 'Low' },
  ];

  const levelColors = {
    'Low':      [5, 150, 105],
    'Moderate': [217, 119, 6],
    'High':     [220, 38, 38],
    'Critical': [124, 45, 18],
  };

  const rfBody = rfactors.map(f => [f.factor, f.value, `${f.score}%`, f.level]);

  doc.autoTable({
    startY: y,
    head: [['Risk Factor', 'Patient Value', 'Risk Score', 'Risk Level']],
    body: rfBody,
    margin: { left: margin, right: margin },
    styles: { fontSize: 9.5, cellPadding: 3.5, lineColor: [226, 232, 240], lineWidth: 0.3 },
    headStyles: { fillColor: [30, 41, 59], textColor: 255, fontStyle: 'bold', fontSize: 10 },
    alternateRowStyles: { fillColor: [248, 250, 252] },
    columnStyles: {
      0: { fontStyle: 'bold', textColor: [30, 58, 138], cellWidth: 48 },
      1: { cellWidth: 55 },
      2: { cellWidth: 28, halign: 'center' },
      3: { cellWidth: 30, halign: 'center', fontStyle: 'bold' }
    },
    didParseCell: (data) => {
      if (data.column.index === 3 && data.section === 'body') {
        const lvl = data.cell.raw;
        const col = levelColors[lvl] || [100, 100, 100];
        data.cell.styles.textColor = col;
      }
      if (data.column.index === 2 && data.section === 'body') {
        const score = parseInt(data.cell.raw);
        if (score >= 75) data.cell.styles.textColor = [220, 38, 38];
        else if (score >= 45) data.cell.styles.textColor = [217, 119, 6];
        else data.cell.styles.textColor = [5, 150, 105];
      }
    },
    didDrawPage: () => drawPageFooter()
  });
  y = doc.lastAutoTable.finalY + 6;

  // ══════════════════════════════════════════════
  // SECTION 3 — PERSONALISED ROADMAP
  // ══════════════════════════════════════════════
  const roadmapSteps = generateRoadmapSteps(input, pct);

  checkPageBreak(30);
  sectionHeader('Personalised Prevention & Recovery Roadmap', String.fromCodePoint(0x1F5FA));

  // Subtitle
  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'italic');
  doc.setTextColor(71, 85, 105);
  const subtitleText = `Personalised plan for ${patientName} — ${roadmapSteps.length} evidence-based steps ordered by clinical urgency. Risk: ${riskLabel} (${pct}%).`;
  const subtitleLines = doc.splitTextToSize(subtitleText, contentW);
  doc.text(subtitleLines, margin, y);
  y += subtitleLines.length * 4 + 5;

  const priorityColorMap = {
    'URGENT':        [220, 38, 38],
    'HIGH PRIORITY': [220, 38, 38],
    'MODERATE':      [217, 119, 6],
    'PREVENTIVE':    [5, 150, 105],
    'MAINTENANCE':   [5, 150, 105],
    'SUPPORTIVE':    [124, 58, 237],
    'AWARENESS':     [220, 38, 38],
  };

  roadmapSteps.forEach((step, idx) => {
    const prioColor = priorityColorMap[step.priority] || [100, 100, 100];

    // Estimate height needed
    const descLines = doc.splitTextToSize(step.desc, contentW - 20);
    const actionsText = step.actions.join('  |  ');
    const actLines = doc.splitTextToSize(actionsText, contentW - 20);
    const estH = 10 + descLines.length * 4.5 + actLines.length * 4.5 + 14;

    checkPageBreak(estH + 6);

    // Step box
    doc.setFillColor(249, 250, 251);
    doc.setDrawColor(226, 232, 240);
    doc.setLineWidth(0.4);
    doc.roundedRect(margin, y, contentW, estH, 2.5, 2.5, 'FD');

    // Left accent strip
    doc.setFillColor(prioColor[0], prioColor[1], prioColor[2]);
    doc.roundedRect(margin, y, 4, estH, 1, 1, 'F');

    let sy = y + 5;

    // Step number + priority badge
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(prioColor[0], prioColor[1], prioColor[2]);
    doc.text(`STEP ${idx + 1}`, margin + 8, sy + 1);

    // Priority badge
    const prioText = step.priority;
    const prioW = doc.getTextWidth(prioText) + 6;
    doc.setFillColor(prioColor[0], prioColor[1], prioColor[2]);
    doc.roundedRect(margin + 28, sy - 3.5, prioW, 6, 1.5, 1.5, 'F');
    doc.setFontSize(7);
    doc.setTextColor(255, 255, 255);
    doc.text(prioText, margin + 28 + prioW / 2, sy + 0.8, { align: 'center' });

    sy += 6;

    // Title
    doc.setFontSize(10.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(15, 23, 42);
    doc.text(step.title, margin + 8, sy);
    sy += 6;

    // Description
    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(55, 65, 81);
    doc.text(descLines, margin + 8, sy);
    sy += descLines.length * 4.5 + 3;

    // Action tags row
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(prioColor[0], prioColor[1], prioColor[2]);
    doc.text('Actions: ', margin + 8, sy);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(55, 65, 81);
    const actFull = step.actions.map(a => `• ${a}`).join('   ');
    const actFLines = doc.splitTextToSize(actFull, contentW - 28);
    doc.text(actFLines, margin + 8, sy + 4);

    y += estH + 4;
  });

  // ══════════════════════════════════════════════
  // DISCLAIMER PAGE
  // ══════════════════════════════════════════════
  checkPageBreak(50);
  y += 6;

  doc.setFillColor(254, 243, 199);
  doc.setDrawColor(217, 119, 6);
  doc.setLineWidth(0.5);
  doc.roundedRect(margin, y, contentW, 36, 3, 3, 'FD');

  doc.setFontSize(10);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(146, 64, 14);
  doc.text('Medical Disclaimer', margin + 5, y + 9);

  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(92, 45, 9);
  const disclaimer = [
    'This AI-generated stroke risk report is intended for informational and educational purposes only.',
    'It does NOT constitute a medical diagnosis, clinical opinion, or substitute for professional healthcare advice.',
    'Always consult a qualified physician, neurologist, or cardiologist for medical guidance.',
    'The MACS-Net model was trained on the Kaggle Stroke Prediction Dataset (5,110 patients).',
    'Risk predictions are statistical estimates and may not apply to individual circumstances.'
  ];
  disclaimer.forEach((line, i) => {
    doc.text(line, margin + 5, y + 17 + i * 4);
  });

  y += 44;

  // Final generation timestamp
  doc.setFontSize(8);
  doc.setFont('helvetica', 'italic');
  doc.setTextColor(148, 163, 184);
  doc.text(`Generated: ${new Date().toLocaleString('en-IN')}  |  Report ID: ${reportId}  |  MACS-Net v1.0`, pageW / 2, y, { align: 'center' });

  // Draw footer on last page
  drawPageFooter();

  // ── Save ────────────────────────────────────────
  const filename = `MACS-Net_Stroke_Report_${patientName.replace(/\s+/g, '_')}_${new Date().toISOString().slice(0,10)}.pdf`;
  doc.save(filename);
}

/* ──────────────────────────────────────────
   9. RESTART
────────────────────────────────────────── */
function restartAssessment() {
  document.getElementById('predictForm').reset();
  document.getElementById('result-section').classList.add('hidden');
  goToStep(1);

  // Reset bmi category
  document.getElementById('bmi_category').textContent = '—';
  document.getElementById('bmi_category').style.color = '';
  document.getElementById('bmi_category').style.background = '';

  // Reset sliders
  document.getElementById('age_slider').value = 45;
  document.getElementById('glucose_slider').value = 106;
  document.getElementById('bmi_slider').value = 28;
  ['age_slider', 'glucose_slider', 'bmi_slider'].forEach(id => updateSliderFill(document.getElementById(id)));

  window.scrollTo({ top: document.getElementById('predictor').offsetTop - 80, behavior: 'smooth' });
}

/* ──────────────────────────────────────────
   10. TOAST NOTIFICATION
────────────────────────────────────────── */
function showToast(message, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'toast';
  const colors = { warning: '#D97706', error: '#DC2626', info: '#2563EB', success: '#059669' };
  toast.style.cssText = `
    position: fixed; bottom: 32px; right: 32px; z-index: 9999;
    background: white; border: 1px solid ${colors[type]}40;
    border-left: 4px solid ${colors[type]};
    color: ${colors[type]}; padding: 14px 20px; border-radius: 12px;
    font-size: 0.875rem; font-weight: 500; font-family: var(--font-body);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    animation: slideInToast 0.3s ease;
    max-width: 360px; line-height: 1.4;
  `;
  toast.textContent = message;

  const style = document.createElement('style');
  style.textContent = `@keyframes slideInToast { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }`;
  document.head.appendChild(style);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

/* ──────────────────────────────────────────
   11. SCROLL & INIT
────────────────────────────────────────── */
window.addEventListener('scroll', () => {
  const navbar = document.getElementById('navbar');
  if (window.scrollY > 10) navbar.classList.add('scrolled');
  else navbar.classList.remove('scrolled');

  // Active nav link
  const sections = ['hero', 'about', 'predictor', 'result-section'];
  let current = '';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && el.getBoundingClientRect().top < 120) current = id;
  });
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === '#' + current);
  });
});

// Intersection Observer for about cards
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animation = `fadeSlideUp 0.5s ${entry.target.dataset.delay || 0}ms ease both`;
    }
  });
}, { threshold: 0.15 });

document.addEventListener('DOMContentLoaded', () => {
  syncSliders();
  document.querySelectorAll('.about-card').forEach(card => observer.observe(card));
});

/* ──────────────────────────────────────────
   12. EMAIL MODAL & SEND REPORT
────────────────────────────────────────── */

const BACKEND_URL = 'http://localhost:8000';

function openEmailModal() {
  if (!lastPredictionInput || !lastPredictionResult) {
    showToast('Please run an assessment first before sending.', 'warning');
    return;
  }

  const pct  = Math.round(lastPredictionResult.prob * 100);
  const name = lastPredictionInput.patient_name || 'Anonymous Patient';

  // Set risk level label
  let riskLabel;
  if (pct < 20)      riskLabel = 'LOW RISK';
  else if (pct < 40) riskLabel = 'MODERATE RISK';
  else if (pct < 60) riskLabel = 'HIGH RISK';
  else               riskLabel = 'CRITICAL RISK';

  // Populate patient preview strip
  document.getElementById('emailAvatar').textContent     = name.charAt(0).toUpperCase();
  document.getElementById('emailPatientName').textContent = name;
  document.getElementById('emailPatientRisk').textContent = `${riskLabel} · ${pct}% Stroke Probability`;

  // Clear previous state
  document.getElementById('recipientEmail').value = '';
  const statusEl = document.getElementById('emailStatus');
  statusEl.style.display = 'none';
  statusEl.className = 'email-status';
  statusEl.textContent = '';

  // Reset button
  const sendBtn = document.getElementById('emailSendBtn');
  sendBtn.disabled = false;
  sendBtn.querySelector('.email-btn-text').style.display = 'flex';
  sendBtn.querySelector('.email-btn-loader').style.display = 'none';

  // Open overlay
  document.getElementById('emailModalOverlay').classList.add('open');
  document.body.style.overflow = 'hidden';

  // Focus input
  setTimeout(() => document.getElementById('recipientEmail').focus(), 350);
}

function closeEmailModal(event) {
  // If called from overlay click, only close if clicking the backdrop (not the modal itself)
  if (event && event.target !== document.getElementById('emailModalOverlay')) return;

  document.getElementById('emailModalOverlay').classList.remove('open');
  document.body.style.overflow = '';
}

async function sendReportByEmail() {
  const emailInput = document.getElementById('recipientEmail');
  const recipient  = emailInput.value.trim();

  // Validate email
  if (!recipient || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(recipient)) {
    emailInput.style.borderColor = '#DC2626';
    emailInput.style.boxShadow   = '0 0 0 4px rgba(220,38,38,0.12)';
    emailInput.focus();
    setTimeout(() => {
      emailInput.style.borderColor = '';
      emailInput.style.boxShadow   = '';
    }, 2000);
    showToast('Please enter a valid email address.', 'warning');
    return;
  }

  const sendBtn   = document.getElementById('emailSendBtn');
  const statusEl  = document.getElementById('emailStatus');
  const pct       = Math.round(lastPredictionResult.prob * 100);
  const patName   = lastPredictionInput.patient_name || 'Anonymous Patient';

  // Risk level
  let riskLabel;
  if (pct < 20)      riskLabel = 'LOW';
  else if (pct < 40) riskLabel = 'MODERATE';
  else if (pct < 60) riskLabel = 'HIGH';
  else               riskLabel = 'CRITICAL';

  // Loading state
  sendBtn.disabled = true;
  sendBtn.querySelector('.email-btn-text').style.display  = 'none';
  sendBtn.querySelector('.email-btn-loader').style.display = 'flex';
  statusEl.style.display  = 'none';
  statusEl.className      = 'email-status';

  try {
    // ── 1. Generate PDF and extract base64 ──────────────
    const pdfBase64 = getPDFBase64(lastPredictionInput, lastPredictionResult);

    // ── 2. POST to FastAPI backend ───────────────────────
    const response = await fetch(`${BACKEND_URL}/api/send-report`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recipient_email: recipient,
        patient_name:    patName,
        risk_level:      riskLabel,
        risk_percentage: pct,
        pdf_base64:      pdfBase64,
        report_date:     new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' })
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `Server error (${response.status})`);
    }

    // ── Success ───────────────────────────────────────────
    statusEl.className   = 'email-status success';
    statusEl.innerHTML   = `✅ Report sent successfully!<br><span style="font-weight:400;font-size:0.8125rem;">Delivered to <strong>${recipient}</strong></span>`;
    statusEl.style.display = 'block';
    showToast(`Report sent to ${recipient}!`, 'success');

    // Auto-close modal after 3.5s
    setTimeout(() => closeEmailModal(), 3500);

  } catch (err) {
    console.error('Email send error:', err);

    let errMsg = err.message || 'Unknown error.';
    if (errMsg.includes('fetch') || errMsg.includes('Failed to fetch') || errMsg.includes('NetworkError')) {
      errMsg = 'Cannot reach the backend server. Please ensure the FastAPI server is running on port 8000 (run start.bat).';
    }

    statusEl.className    = 'email-status error';
    statusEl.innerHTML    = `❌ Failed to send email.<br><span style="font-weight:400;font-size:0.8125rem;">${errMsg}</span>`;
    statusEl.style.display = 'block';

  } finally {
    sendBtn.disabled = false;
    sendBtn.querySelector('.email-btn-text').style.display  = 'flex';
    sendBtn.querySelector('.email-btn-loader').style.display = 'none';
  }
}

/**
 * Generate PDF using jsPDF and return the raw base64 string
 * (strips the "data:application/pdf;base64," prefix).
 */
function getPDFBase64(input, result) {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

  const pageW    = 210;
  const pageH    = 297;
  const margin   = 14;
  const contentW = pageW - margin * 2;
  let y = 0;

  const pct         = Math.round(result.prob * 100);
  const patientName = input.patient_name || 'Anonymous Patient';
  const assessDate  = new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' });
  const reportId    = 'MN-' + Date.now().toString(36).toUpperCase();

  let riskLabel, riskR, riskG, riskB;
  if (pct < 20)      { riskLabel = 'LOW RISK';      riskR=5;   riskG=150; riskB=105; }
  else if (pct < 40) { riskLabel = 'MODERATE RISK'; riskR=217; riskG=119; riskB=6;   }
  else if (pct < 60) { riskLabel = 'HIGH RISK';     riskR=220; riskG=38;  riskB=38;  }
  else               { riskLabel = 'CRITICAL RISK';  riskR=124; riskG=45;  riskB=18;  }

  function checkPageBreak(needed = 20) {
    if (y + needed > pageH - 20) { doc.addPage(); y = margin + 5; drawPageFooter(); }
  }
  function drawPageFooter() {
    const pg = doc.internal.getCurrentPageInfo().pageNumber;
    doc.setFontSize(7.5); doc.setTextColor(150,150,150); doc.setFont('helvetica','italic');
    doc.text('MACS-Net Stroke Risk Intelligence Platform  |  For informational purposes only.', margin, pageH - 8);
    doc.text(`Page ${pg}`, pageW - margin, pageH - 8, { align: 'right' });
  }
  function sectionHeader(title) {
    checkPageBreak(16); y += 5;
    doc.setFillColor(240,245,255); doc.roundedRect(margin, y, contentW, 9, 2, 2, 'F');
    doc.setFontSize(10); doc.setFont('helvetica','bold'); doc.setTextColor(30,58,138);
    doc.text(title, margin + 4, y + 6); y += 13;
  }

  // Header band
  doc.setFillColor(15,23,42); doc.rect(0,0,pageW,42,'F');
  doc.setFillColor(37,99,235); doc.rect(0,38,pageW,4,'F');
  doc.setFontSize(20); doc.setFont('helvetica','bold');
  doc.setTextColor(255,255,255); doc.text('MACS', margin, 18);
  doc.setTextColor(99,179,237); doc.text('Net', margin+28, 18);
  doc.setFontSize(9); doc.setFont('helvetica','normal');
  doc.setTextColor(148,163,184); doc.text('Multi-modal Attention Clinical Stroke Network', margin, 26);
  doc.text('Stroke Risk Assessment Report', margin, 33);
  doc.setFontSize(8.5); doc.setTextColor(203,213,225);
  doc.text(`Report ID: ${reportId}`, pageW-margin, 18, {align:'right'});
  doc.text(`Date: ${assessDate}`, pageW-margin, 26, {align:'right'});
  doc.text('Confidential Medical Document', pageW-margin, 33, {align:'right'});
  y = 50;

  // Patient banner
  doc.setFillColor(248,250,252); doc.setDrawColor(226,232,240); doc.setLineWidth(0.4);
  doc.roundedRect(margin, y, contentW, 24, 3, 3, 'FD');
  doc.setFillColor(37,99,235); doc.circle(margin+12, y+12, 9, 'F');
  doc.setFontSize(12); doc.setFont('helvetica','bold'); doc.setTextColor(255,255,255);
  doc.text(patientName.charAt(0).toUpperCase(), margin+12, y+16, {align:'center'});
  doc.setFontSize(13); doc.setFont('helvetica','bold'); doc.setTextColor(15,23,42);
  doc.text(patientName, margin+26, y+11);
  doc.setFontSize(8.5); doc.setFont('helvetica','normal'); doc.setTextColor(100,116,139);
  doc.text(`Assessment Date: ${assessDate}`, margin+26, y+19);
  doc.setFillColor(riskR,riskG,riskB); doc.roundedRect(pageW-margin-40, y+5, 38, 14, 3, 3, 'F');
  doc.setFontSize(9); doc.setFont('helvetica','bold'); doc.setTextColor(255,255,255);
  doc.text(riskLabel, pageW-margin-21, y+14, {align:'center'});
  y += 32;

  // Risk score box
  doc.setFillColor(riskR,riskG,riskB); doc.roundedRect(margin, y, contentW, 32, 4, 4, 'F');
  doc.setFontSize(42); doc.setFont('helvetica','bold'); doc.setTextColor(255,255,255);
  doc.text(`${pct}%`, pageW/2, y+22, {align:'center'});
  doc.setFontSize(10); doc.setFont('helvetica','normal'); doc.setTextColor(220,230,245);
  doc.text('Stroke Risk Probability', pageW/2, y+30, {align:'center'});
  doc.setTextColor(255,255,255);
  y += 38;

  doc.setFontSize(8); doc.setFont('helvetica','italic'); doc.setTextColor(100,116,139);
  const mLine = doc.splitTextToSize('Model: MACS-Net CFE (RF + GB Ensemble) · 5,110 patients · SMOTE · 5-fold CV · Threshold: 35%', contentW);
  doc.text(mLine, pageW/2, y, {align:'center'}); y += mLine.length * 4 + 4;

  // Clinical profile
  sectionHeader('Clinical Profile Summary');
  const profileRows = [
    ['Age', `${input.age} years`], ['Gender', input.gender],
    ['Marital Status', input.ever_married === 'Yes' ? 'Married / Previously Married' : 'Never Married'],
    ['Work Type', input.work_type.replace(/_/g,' ')], ['Residence Type', input.Residence_type],
    ['Avg. Blood Glucose', `${input.avg_glucose_level} mg/dL`],
    ['BMI', `${parseFloat(input.bmi).toFixed(1)} kg/m²`],
    ['Hypertension', input.hypertension ? 'Yes (Diagnosed)' : 'No'],
    ['Heart Disease', input.heart_disease ? 'Yes (Diagnosed)' : 'No'],
    ['Smoking Status', input.smoking_status.charAt(0).toUpperCase() + input.smoking_status.slice(1)],
  ];
  doc.autoTable({ startY: y, head:[['Clinical Parameter','Patient Value']], body: profileRows,
    margin:{left:margin,right:margin}, styles:{fontSize:9.5,cellPadding:3.5,lineColor:[226,232,240],lineWidth:0.3},
    headStyles:{fillColor:[37,99,235],textColor:255,fontStyle:'bold',fontSize:10},
    alternateRowStyles:{fillColor:[248,250,252]},
    columnStyles:{0:{fontStyle:'bold',textColor:[30,58,138],cellWidth:68},1:{textColor:[15,23,42]}},
    didDrawPage:()=>drawPageFooter() });
  y = doc.lastAutoTable.finalY + 6;

  // Risk factors
  sectionHeader('Risk Factor Analysis');
  const levelColors = {'Low':[5,150,105],'Moderate':[217,119,6],'High':[220,38,38],'Critical':[124,45,18]};
  const rfactors = [
    {f:'Age',v:`${input.age} yrs`,s:Math.min(100,Math.round((input.age/82)*100)),l:input.age>70?'Critical':input.age>55?'High':input.age>40?'Moderate':'Low'},
    {f:'Blood Glucose',v:`${input.avg_glucose_level} mg/dL`,s:Math.min(100,Math.round(((input.avg_glucose_level-50)/250)*100)),l:input.avg_glucose_level>200?'Critical':input.avg_glucose_level>125?'High':input.avg_glucose_level>100?'Moderate':'Low'},
    {f:'BMI',v:`${parseFloat(input.bmi).toFixed(1)} kg/m²`,s:Math.min(100,Math.round(((input.bmi-10)/50)*100)),l:input.bmi>=40?'Critical':input.bmi>=30?'High':input.bmi>=25?'Moderate':'Low'},
    {f:'Hypertension',v:input.hypertension?'Yes':'No',s:input.hypertension?80:10,l:input.hypertension?'High':'Low'},
    {f:'Heart Disease',v:input.heart_disease?'Yes':'No',s:input.heart_disease?75:10,l:input.heart_disease?'High':'Low'},
    {f:'Smoking',v:input.smoking_status.charAt(0).toUpperCase()+input.smoking_status.slice(1),s:input.smoking_status==='smokes'?85:input.smoking_status==='formerly smoked'?45:10,l:input.smoking_status==='smokes'?'High':input.smoking_status==='formerly smoked'?'Moderate':'Low'},
    {f:'Overall AI Risk',v:`${pct}%`,s:pct,l:pct>=60?'Critical':pct>=40?'High':pct>=20?'Moderate':'Low'},
  ];
  doc.autoTable({ startY:y, head:[['Risk Factor','Value','Score','Level']], body:rfactors.map(r=>[r.f,r.v,`${r.s}%`,r.l]),
    margin:{left:margin,right:margin}, styles:{fontSize:9.5,cellPadding:3.5,lineColor:[226,232,240],lineWidth:0.3},
    headStyles:{fillColor:[30,41,59],textColor:255,fontStyle:'bold',fontSize:10},
    alternateRowStyles:{fillColor:[248,250,252]},
    columnStyles:{0:{fontStyle:'bold',textColor:[30,58,138],cellWidth:48},1:{cellWidth:55},2:{cellWidth:28,halign:'center'},3:{cellWidth:30,halign:'center',fontStyle:'bold'}},
    didParseCell:(data)=>{
      if(data.column.index===3&&data.section==='body'){const c=levelColors[data.cell.raw]||[100,100,100];data.cell.styles.textColor=c;}
      if(data.column.index===2&&data.section==='body'){const s=parseInt(data.cell.raw);data.cell.styles.textColor=s>=75?[220,38,38]:s>=45?[217,119,6]:[5,150,105];}
    }, didDrawPage:()=>drawPageFooter() });
  y = doc.lastAutoTable.finalY + 6;

  // Roadmap
  const roadmapSteps = generateRoadmapSteps(input, pct);
  checkPageBreak(30); sectionHeader('Personalised Prevention & Recovery Roadmap');
  doc.setFontSize(8.5); doc.setFont('helvetica','italic'); doc.setTextColor(71,85,105);
  const sub = doc.splitTextToSize(`Personalised plan for ${patientName} — ${roadmapSteps.length} evidence-based steps. Risk: ${riskLabel} (${pct}%).`, contentW);
  doc.text(sub, margin, y); y += sub.length * 4 + 5;

  const prioColors = {'URGENT':[220,38,38],'HIGH PRIORITY':[220,38,38],'MODERATE':[217,119,6],'PREVENTIVE':[5,150,105],'MAINTENANCE':[5,150,105],'SUPPORTIVE':[124,58,237],'AWARENESS':[220,38,38]};
  roadmapSteps.forEach((step, idx) => {
    const pc = prioColors[step.priority] || [100,100,100];
    const descL = doc.splitTextToSize(step.desc, contentW-20);
    const actL  = doc.splitTextToSize(step.actions.map(a=>`• ${a}`).join('   '), contentW-28);
    const estH  = 10 + descL.length*4.5 + actL.length*4.5 + 14;
    checkPageBreak(estH+6);
    doc.setFillColor(249,250,251); doc.setDrawColor(226,232,240); doc.setLineWidth(0.4);
    doc.roundedRect(margin, y, contentW, estH, 2.5, 2.5, 'FD');
    doc.setFillColor(pc[0],pc[1],pc[2]); doc.roundedRect(margin, y, 4, estH, 1, 1, 'F');
    let sy = y+5;
    doc.setFontSize(8); doc.setFont('helvetica','bold'); doc.setTextColor(pc[0],pc[1],pc[2]);
    doc.text(`STEP ${idx+1}`, margin+8, sy+1);
    const prioW = doc.getTextWidth(step.priority)+6;
    doc.setFillColor(pc[0],pc[1],pc[2]); doc.roundedRect(margin+28, sy-3.5, prioW, 6, 1.5, 1.5, 'F');
    doc.setFontSize(7); doc.setTextColor(255,255,255);
    doc.text(step.priority, margin+28+prioW/2, sy+0.8, {align:'center'});
    sy += 6;
    doc.setFontSize(10.5); doc.setFont('helvetica','bold'); doc.setTextColor(15,23,42);
    doc.text(step.title, margin+8, sy); sy += 6;
    doc.setFontSize(8.5); doc.setFont('helvetica','normal'); doc.setTextColor(55,65,81);
    doc.text(descL, margin+8, sy); sy += descL.length*4.5+3;
    doc.setFontSize(7.5); doc.setFont('helvetica','bold'); doc.setTextColor(pc[0],pc[1],pc[2]);
    doc.text('Actions: ', margin+8, sy);
    doc.setFont('helvetica','normal'); doc.setTextColor(55,65,81);
    doc.text(actL, margin+8, sy+4);
    y += estH+4;
  });

  // Disclaimer
  checkPageBreak(50); y += 6;
  doc.setFillColor(254,243,199); doc.setDrawColor(217,119,6); doc.setLineWidth(0.5);
  doc.roundedRect(margin, y, contentW, 36, 3, 3, 'FD');
  doc.setFontSize(10); doc.setFont('helvetica','bold'); doc.setTextColor(146,64,14);
  doc.text('Medical Disclaimer', margin+5, y+9);
  doc.setFontSize(8.5); doc.setFont('helvetica','normal'); doc.setTextColor(92,45,9);
  ['This AI report is for informational purposes only.',
   'It does NOT constitute a medical diagnosis or substitute for professional advice.',
   'Always consult a qualified physician, neurologist, or cardiologist.',
   'MACS-Net model trained on Kaggle Stroke Prediction Dataset (5,110 patients).',
   'Risk predictions are statistical estimates — individual cases may vary.'
  ].forEach((l,i) => doc.text(l, margin+5, y+17+i*4));
  y += 44;
  doc.setFontSize(8); doc.setFont('helvetica','italic'); doc.setTextColor(148,163,184);
  doc.text(`Generated: ${new Date().toLocaleString('en-IN')}  |  Report ID: ${reportId}  |  MACS-Net v1.0`, pageW/2, y, {align:'center'});
  drawPageFooter();

  // Return raw base64 (strip "data:application/pdf;base64,")
  const dataUri = doc.output('datauristring');
  return dataUri.split(',')[1];
}

