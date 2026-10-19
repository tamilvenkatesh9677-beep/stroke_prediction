/* ═══════════════════════════════════════════
   app.js — MACS-Net Stroke Risk Intelligence
   Production-grade frontend interface
   Connects to FastAPI backend REST API
═══════════════════════════════════════════ */

const API_BASE = window.location.origin.includes('localhost') || 
                 window.location.origin.includes('127.0.0.1') ||
                 window.location.protocol === 'file:'
  ? 'http://localhost:8000'
  : window.location.origin;

let currentStep = 1;
let debounceTimeout = null;
let apiOnline = false;

// ── DOM Elements ──────────────────────────────────────────────────────────
const dom = {
  form:            document.getElementById('macsForm'),
  submitBtn:       document.getElementById('submitBtn'),
  apiStatus:       document.getElementById('apiStatus'),
  apiRecall:       document.getElementById('apiRecall'),
  toast:           document.getElementById('toast'),
  
  // Steps
  steps:           document.querySelectorAll('.wiz-step'),
  tabs:            document.querySelectorAll('.step-tab'),
  wpFill:          document.getElementById('wpFill'),
  summaryPreview:  document.getElementById('summaryPreview'),
  spGrid:          document.getElementById('spGrid'),

  // Form Inputs
  fName:           document.getElementById('f_name'),
  fAge:            document.getElementById('f_age'),
  slAge:           document.getElementById('sl_age'),
  fGender:         document.getElementById('f_gender'),
  fMarried:        document.getElementById('f_married'),
  fWork:           document.getElementById('f_work'),
  fResidence:      document.getElementById('f_residence'),
  fHtn:            document.getElementById('f_htn'),
  fHd:             document.getElementById('f_hd'),
  fGlucose:        document.getElementById('f_glucose'),
  slGlucose:       document.getElementById('sl_glucose'),
  fBmi:            document.getElementById('f_bmi'),
  slBmi:           document.getElementById('sl_bmi'),
  bmiPill:         document.getElementById('bmiPill'),

  // Live Banner
  liveBanner:      document.getElementById('liveBanner'),
  livePct:         document.getElementById('livePct'),
  liveBar:         document.getElementById('liveBar'),
  liveLevel:       document.getElementById('liveLevel'),
  heroRisk:        document.getElementById('heroRisk'),

  // Results Section
  resultSection:   document.getElementById('result-section'),
  rhPatient:       document.getElementById('rh_patient'),
  rhDate:          document.getElementById('rh_date'),
  rhBadge:         document.getElementById('rh_badge'),
  gaugeArc:        document.getElementById('gaugeArc'),
  gaugeNeedle:     document.getElementById('gaugeNeedle'),
  gaugePct:        document.getElementById('gaugePct'),
  gaugeInf:        document.getElementById('gaugeInf'),
  factorsList:     document.getElementById('factorsList'),
  profileList:     document.getElementById('profileList'),
  barList:         document.getElementById('barList'),
  modelInfoCard:   document.getElementById('modelInfoCard'),
  rmSub:           document.getElementById('rmSub'),
  rmSteps:         document.getElementById('rmSteps'),
};

// ── Startup & Initialization ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initSliderSync();
  initFormListeners();
  checkApiHealth();
  
  // Set initial BMI indicator
  updateBmiIndicator(28.0);
});

// ── API Health Check & Stats ──────────────────────────────────────────────
async function checkApiHealth() {
  const statusDot = dom.apiStatus.querySelector('.status-dot');
  const statusLabel = dom.apiStatus.querySelector('.status-label');

  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      apiOnline = true;
      dom.apiStatus.classList.add('online');
      dom.apiStatus.classList.remove('offline');
      statusLabel.textContent = 'API Connected';
      
      // Update recall rate from model info if available
      if (data.cv_recall) {
        const recallPct = Math.round(data.cv_recall * 100);
        if (dom.apiRecall) dom.apiRecall.textContent = `~${recallPct}%`;
      }
      
      // Fetch stats
      fetchStats();
    } else {
      throw new Error();
    }
  } catch (err) {
    apiOnline = false;
    dom.apiStatus.classList.remove('online');
    dom.apiStatus.classList.add('offline');
    statusLabel.textContent = 'Offline Mode';
    showToast('FastAPI backend offline. Live prediction preview disabled.', 'warning');
  }
}

async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (res.ok) {
      const stats = await res.json();
      // Update stats dynamically in the Hero metrics if they exist
      const patientCountEl = document.querySelector('.hm-item:nth-child(1) .hm-val');
      if (patientCountEl) {
        animateValue(patientCountEl, 0, stats.total_patients, 1500);
      }
    }
  } catch (err) {
    console.error('Error fetching stats:', err);
  }
}

function animateValue(obj, start, end, duration) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    obj.innerHTML = Math.floor(progress * (end - start) + start).toLocaleString();
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}

// ── Slider Sync ──────────────────────────────────────────────────────────
function initSliderSync() {
  const sync = (input, slider, onInputFn) => {
    input.addEventListener('input', () => {
      slider.value = input.value;
      updateSliderBackground(slider);
      if (onInputFn) onInputFn(parseFloat(input.value));
    });
    slider.addEventListener('input', () => {
      input.value = slider.value;
      updateSliderBackground(slider);
      if (onInputFn) onInputFn(parseFloat(slider.value));
    });
    // Set initial layout
    updateSliderBackground(slider);
  };

  sync(dom.fAge, dom.slAge);
  sync(dom.fGlucose, dom.slGlucose);
  sync(dom.fBmi, dom.slBmi, updateBmiIndicator);
}

function updateSliderBackground(slider) {
  const min = parseFloat(slider.min) || 0;
  const max = parseFloat(slider.max) || 100;
  const val = parseFloat(slider.value) || 0;
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(to right, #2563EB ${pct}%, #E2E8F0 ${pct}%)`;
}

function updateBmiIndicator(bmi) {
  if (!dom.bmiPill) return;
  
  let label = '';
  let cssClass = '';
  
  if (bmi < 18.5) {
    label = 'Underweight';
    cssClass = 'bmi-under';
  } else if (bmi < 25) {
    label = 'Normal Weight';
    cssClass = 'bmi-normal';
  } else if (bmi < 30) {
    label = 'Overweight';
    cssClass = 'bmi-over';
  } else if (bmi < 35) {
    label = 'Obese Class I';
    cssClass = 'bmi-obese-1';
  } else {
    label = 'Obese Class II+';
    cssClass = 'bmi-obese-2';
  }
  
  dom.bmiPill.textContent = `${bmi.toFixed(1)} — ${label}`;
  dom.bmiPill.className = `bmi-pill ${cssClass}`;
}

// ── Step Navigation ────────────────────────────────────────────────────────
window.gotoStep = function(step) {
  if (step > currentStep) {
    // Validate current step before advancing
    if (!validateStep(currentStep)) return;
  }

  // Update tabs
  dom.tabs.forEach(tab => {
    const tabNum = parseInt(tab.dataset.tab);
    if (tabNum === step) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Update steps visibility
  dom.steps.forEach(stepEl => {
    const stepNum = parseInt(stepEl.id.replace('step', ''));
    if (stepNum === step) {
      stepEl.classList.add('active');
    } else {
      stepEl.classList.remove('active');
    }
  });

  // Update progress bar
  const pct = step === 1 ? 33 : step === 2 ? 66 : 100;
  dom.wpFill.style.width = `${pct}%`;

  currentStep = step;

  // Build Summary Preview in Step 3
  if (step === 3) {
    buildSummaryPreview();
  }

  // Scroll wizard container to top
  document.getElementById('predictor').scrollIntoView({ behavior: 'smooth', block: 'start' });
};

function validateStep(step) {
  if (step === 1) {
    const age = parseFloat(dom.fAge.value);
    const gender = dom.fGender.value;
    const married = dom.fMarried.value;
    const work = dom.fWork.value;
    const residence = dom.fResidence.value;

    if (isNaN(age) || age < 1 || age > 100) {
      showToast('Please enter a valid age between 1 and 100.', 'warning');
      dom.fAge.focus();
      return false;
    }
    if (!gender) {
      showToast('Please select patient gender.', 'warning');
      dom.fGender.focus();
      return false;
    }
    if (!married) {
      showToast('Please select marital status.', 'warning');
      dom.fMarried.focus();
      return false;
    }
    if (!work) {
      showToast('Please select work type.', 'warning');
      dom.fWork.focus();
      return false;
    }
    if (!residence) {
      showToast('Please select residence type.', 'warning');
      dom.fResidence.focus();
      return false;
    }
  } else if (step === 2) {
    const glucose = parseFloat(dom.fGlucose.value);
    const bmi = parseFloat(dom.fBmi.value);

    if (isNaN(glucose) || glucose < 50 || glucose > 300) {
      showToast('Please enter a valid blood glucose level (50–300 mg/dL).', 'warning');
      dom.fGlucose.focus();
      return false;
    }
    if (isNaN(bmi) || bmi < 10 || bmi > 100) {
      showToast('Please enter a valid BMI (10–100 kg/m²).', 'warning');
      dom.fBmi.focus();
      return false;
    }
  }
  return true;
}

function buildSummaryPreview() {
  const getSmokingText = () => {
    const checked = document.querySelector('input[name="smoking"]:checked');
    return checked ? checked.nextElementSibling.querySelector('span:last-child').textContent : 'Not Selected';
  };

  const gridData = [
    { label: 'Name', value: dom.fName.value.trim() || 'Anonymous' },
    { label: 'Age', value: `${dom.fAge.value} yrs` },
    { label: 'Gender', value: dom.fGender.value },
    { label: 'Residence', value: dom.fResidence.value },
    { label: 'Hypertension', value: dom.fHtn.checked ? 'Yes 🔴' : 'No 🟢' },
    { label: 'Heart Disease', value: dom.fHd.checked ? 'Yes 🔴' : 'No 🟢' },
    { label: 'Glucose Level', value: `${dom.fGlucose.value} mg/dL` },
    { label: 'BMI', value: `${dom.fBmi.value} kg/m²` },
    { label: 'Smoking Status', value: getSmokingText() }
  ];

  dom.spGrid.innerHTML = gridData.map(item => `
    <div class="sp-item">
      <small>${item.label}</small>
      <strong>${item.value}</strong>
    </div>
  `).join('');
}

// ── Debounced Real-time Preview ───────────────────────────────────────────
function initFormListeners() {
  const inputs = [
    dom.fAge, dom.slAge, dom.fGender, dom.fMarried, dom.fWork, dom.fResidence,
    dom.fHtn, dom.fHd, dom.fGlucose, dom.slGlucose, dom.fBmi, dom.slBmi
  ];

  inputs.forEach(input => {
    input.addEventListener('change', triggerLivePrediction);
    input.addEventListener('input', triggerLivePrediction);
  });

  // Smoking radios
  document.querySelectorAll('input[name="smoking"]').forEach(radio => {
    radio.addEventListener('change', triggerLivePrediction);
  });

  // Handle form submission
  dom.form.addEventListener('submit', handleFormSubmit);
}

function triggerLivePrediction() {
  if (!apiOnline) return;

  // Debounce API calls
  clearTimeout(debounceTimeout);
  debounceTimeout = setTimeout(async () => {
    // Check if step 1 features are valid
    const age = parseFloat(dom.fAge.value);
    const gender = dom.fGender.value;
    const married = dom.fMarried.value;
    const work = dom.fWork.value;
    const residence = dom.fResidence.value;
    const glucose = parseFloat(dom.fGlucose.value);
    const bmi = parseFloat(dom.fBmi.value);
    const smokingEl = document.querySelector('input[name="smoking"]:checked');

    // Only hit API if we have at least these key fields filled out
    if (isNaN(age) || !gender || !married || !work || !residence || isNaN(glucose) || isNaN(bmi)) {
      return;
    }

    const payload = {
      patient_name:       dom.fName.value.trim() || 'Anonymous Patient',
      age:                age,
      gender:             gender,
      hypertension:       dom.fHtn.checked ? 1 : 0,
      heart_disease:      dom.fHd.checked ? 1 : 0,
      ever_married:       married,
      work_type:          work,
      Residence_type:     residence,
      avg_glucose_level:  glucose,
      bmi:                bmi,
      smoking_status:     smokingEl ? smokingEl.value : 'Unknown'
    };

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const data = await res.json();
        updateLiveBanner(data);
      }
    } catch (err) {
      console.error('Error fetching live prediction:', err);
    }
  }, 400);
}

function updateLiveBanner(data) {
  dom.liveBanner.classList.remove('hidden');
  dom.livePct.textContent = `${data.risk_percentage}%`;
  dom.liveBar.style.width = `${data.risk_percentage}%`;
  dom.liveBar.style.backgroundColor = data.risk_color;
  dom.liveLevel.textContent = data.risk_level;
  dom.liveLevel.style.color = data.risk_color;

  if (dom.heroRisk) {
    dom.heroRisk.textContent = `${data.risk_percentage}%`;
    dom.heroRisk.style.color = data.risk_color;
  }
}

// ── Form Submission ───────────────────────────────────────────────────────
async function handleFormSubmit(e) {
  e.preventDefault();

  if (!validateStep(1) || !validateStep(2)) {
    return;
  }

  const smokingEl = document.querySelector('input[name="smoking"]:checked');
  if (!smokingEl) {
    showToast('Please select smoking status.', 'warning');
    return;
  }

  // Show loader
  dom.submitBtn.disabled = true;
  dom.submitBtn.querySelector('.sub-text').style.display = 'none';
  dom.submitBtn.querySelector('.sub-loading').style.display = 'flex';

  const payload = {
    patient_name:       dom.fName.value.trim() || 'Anonymous Patient',
    age:                parseFloat(dom.fAge.value),
    gender:             dom.fGender.value,
    hypertension:       dom.fHtn.checked ? 1 : 0,
    heart_disease:      dom.fHd.checked ? 1 : 0,
    ever_married:       dom.fMarried.value,
    work_type:          dom.fWork.value,
    Residence_type:     dom.fResidence.value,
    avg_glucose_level:  parseFloat(dom.fGlucose.value),
    bmi:                parseFloat(dom.fBmi.value),
    smoking_status:     smokingEl.value
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      renderFullResults(payload, data);
      showToast('Stroke Risk Assessment complete!', 'success');
      
      // Scroll to result section
      setTimeout(() => {
        dom.resultSection.classList.remove('hidden');
        dom.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 300);
    } else {
      const errorData = await res.json();
      throw new Error(errorData.detail || 'Prediction failed');
    }
  } catch (err) {
    console.error('Submit error:', err);
    showToast(err.message || 'Server error. Failed to run prediction.', 'error');
  } finally {
    // Hide loader
    dom.submitBtn.disabled = false;
    dom.submitBtn.querySelector('.sub-text').style.display = 'flex';
    dom.submitBtn.querySelector('.sub-loading').style.display = 'none';
  }
}

// ── Rendering Results ─────────────────────────────────────────────────────
function renderFullResults(input, data) {
  // 1. Report Header Info
  dom.rhPatient.textContent = `Patient: ${data.patient_name}`;
  dom.rhDate.textContent = `Date: ${data.timestamp}`;
  
  dom.rhBadge.textContent = `${data.risk_level} RISK`;
  dom.rhBadge.style.backgroundColor = data.risk_color;
  dom.rhBadge.style.color = '#FFFFFF';

  // 2. Gauge Animation
  const pct = data.risk_percentage;
  dom.gaugePct.textContent = `${pct}%`;
  dom.gaugeInf.textContent = `${data.risk_level} Risk Level`;
  dom.gaugeInf.style.color = data.risk_color;

  // Arc length calculations (Full path is 276.5)
  const offset = 276.5 - (pct / 100) * 276.5;
  dom.gaugeArc.style.strokeDashoffset = offset;

  // Needle rotation: from -90deg (0%) to +90deg (100%)
  const deg = (pct / 100) * 180 - 90;
  dom.gaugeNeedle.setAttribute('transform', `rotate(${deg} 110 118)`);

  // 3. Risk Factor Analysis List
  dom.factorsList.innerHTML = data.factor_scores.map(f => `
    <div class="factor-item">
      <div class="f-bullet" style="background-color: ${f.color}"></div>
      <div class="f-info">
        <div class="f-name">${f.factor}</div>
        <div class="f-desc">${f.value}</div>
      </div>
      <div class="f-score">
        <span class="f-badge" style="background-color: ${f.color}15; color: ${f.color}">${f.level.toUpperCase()}</span>
      </div>
    </div>
  `).join('');

  // 4. Clinical Profile details
  const profileItems = [
    { label: 'Patient Name', val: input.patient_name },
    { label: 'Age / Gender', val: `${input.age} yrs / ${input.gender}` },
    { label: 'Hypertension', val: input.hypertension ? 'Hypertensive' : 'Normal BP' },
    { label: 'Heart Disease', val: input.heart_disease ? 'Present' : 'None Detected' },
    { label: 'Blood Glucose', val: `${input.avg_glucose_level} mg/dL` },
    { label: 'BMI Category', val: `${input.bmi} kg/m²` },
    { label: 'Smoking Status', val: input.smoking_status.toUpperCase() },
    { label: 'Work / Residence', val: `${input.work_type} / ${input.Residence_type}` }
  ];

  dom.profileList.innerHTML = profileItems.map(item => `
    <div class="profile-item">
      <span>${item.label}</span>
      <strong>${item.val}</strong>
    </div>
  `).join('');

  // 5. Risk Bars
  dom.barList.innerHTML = data.factor_scores.map(f => `
    <div class="bar-item">
      <div class="bar-lbl">
        <span>${f.factor}</span>
        <strong>${f.score > 0 ? '+' : ''}${Math.round(f.score * 100)}%</strong>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${Math.max(1, Math.min(100, Math.round(Math.abs(f.score) * 100)))}%; background-color: ${f.color}"></div>
      </div>
    </div>
  `).join('');

  // 6. Model Info
  dom.modelInfoCard.innerHTML = `
    <div class="mi-grid">
      <div class="mi-item">
        <small>AI Architecture</small>
        <strong>${data.model_info.name}</strong>
      </div>
      <div class="mi-item">
        <small>Model Type</small>
        <strong>${data.model_info.type}</strong>
      </div>
      <div class="mi-item">
        <small>Validation Method</small>
        <strong>${data.model_info.training}</strong>
      </div>
      <div class="mi-item">
        <small>Performance (ROC-AUC)</small>
        <strong>${data.model_info.cv_roc_auc ? (data.model_info.cv_roc_auc * 100).toFixed(1) + '%' : 'N/A'}</strong>
      </div>
    </div>
  `;

  // 7. Roadmap
  dom.rmSub.textContent = `Prevention and therapeutic actions based on ${data.factor_scores.length} identified risk factors.`;
  dom.rmSteps.innerHTML = data.roadmap.map(step => `
    <div class="rm-step">
      <div class="rms-badge" style="background-color: ${getPriorityColor(step.priority)}20; color: ${getPriorityColor(step.priority)}">
        ${step.priority}
      </div>
      <div class="rms-header">
        <div class="rms-icon">${step.icon}</div>
        <div class="rms-title">
          <span>Step ${step.step_number}</span>
          <h4>${step.title}</h4>
        </div>
      </div>
      <p class="rms-desc">${step.description}</p>
      <div class="rms-actions">
        <strong>Prescribed Actions:</strong>
        <ul>
          ${step.actions.map(action => `<li>${action}</li>`).join('')}
        </ul>
      </div>
    </div>
  `).join('');
}

function getPriorityColor(priority) {
  switch (priority) {
    case 'URGENT':
      return '#DC2626'; // Red
    case 'HIGH PRIORITY':
      return '#D97706'; // Amber
    case 'MODERATE':
      return '#2563EB'; // Blue
    case 'PREVENTIVE':
      return '#059669'; // Green
    case 'MAINTENANCE':
      return '#10B981'; // Emerald
    case 'AWARENESS':
      return '#6366F1'; // Indigo
    default:
      return '#64748B'; // Slate
  }
}

// ── Toast Notifications ────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  if (!dom.toast) return;
  
  dom.toast.textContent = message;
  dom.toast.className = `toast-el show ${type}`;
  
  setTimeout(() => {
    dom.toast.classList.remove('show');
  }, 4000);
}

// ── Action Buttons ────────────────────────────────────────────────────────
window.downloadPDF = function() {
  const element = document.getElementById('pdf-report');
  const patientName = dom.fName.value.trim() || 'Anonymous';
  const cleanName = patientName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
  
  const opt = {
    margin:       [10, 10, 10, 10],
    filename:     `macs_net_stroke_report_${cleanName}.pdf`,
    image:        { type: 'jpeg', quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, letterRendering: true },
    jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };
  
  // Disable button to prevent double-clicks
  const btn = document.getElementById('btnPdf');
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '⚡ Generating PDF...';
  
  html2pdf().set(opt).from(element).save().then(() => {
    btn.disabled = false;
    btn.innerHTML = originalText;
    showToast('PDF Report downloaded successfully.', 'success');
  }).catch(err => {
    console.error('PDF error:', err);
    btn.disabled = false;
    btn.innerHTML = originalText;
    showToast('Failed to download PDF.', 'error');
  });
};

window.shareReport = function() {
  if (navigator.share) {
    navigator.share({
      title: 'MACS-Net Stroke Risk Assessment',
      text: `Completed a stroke risk assessment using the MACS-Net AI Platform.`,
      url: window.location.href
    }).catch(err => console.log('Share error:', err));
  } else {
    // Copy URL to clipboard
    navigator.clipboard.writeText(window.location.href).then(() => {
      showToast('Link copied to clipboard!', 'success');
    }).catch(() => {
      showToast('Sharing not supported on this browser.', 'warning');
    });
  }
};

window.restartApp = function() {
  dom.form.reset();
  
  // Set default values in inputs & ranges
  dom.fAge.value = 45;
  dom.slAge.value = 45;
  updateSliderBackground(dom.slAge);
  
  dom.fGlucose.value = 106;
  dom.slGlucose.value = 106;
  updateSliderBackground(dom.slGlucose);
  
  dom.fBmi.value = 28.0;
  dom.slBmi.value = 28.0;
  updateSliderBackground(dom.slBmi);
  updateBmiIndicator(28.0);
  
  // Hide live banner & reset hero risk
  dom.liveBanner.classList.add('hidden');
  if (dom.heroRisk) dom.heroRisk.textContent = '—';
  
  // Hide result section
  dom.resultSection.classList.add('hidden');
  
  // Go to step 1
  gotoStep(1);
  
  // Scroll to predictor
  document.getElementById('predictor').scrollIntoView({ behavior: 'smooth', block: 'start' });
};
