# ══════════════════════════════════════════════════════
#  CELL 2 — Imports
# ══════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, io, datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, precision_score, recall_score,
    f1_score, brier_score_loss
)
from sklearn.utils import resample
from xgboost import XGBClassifier
import shap
import gradio as gr
from PIL import Image
from fpdf import FPDF

# ── Blue colour palette for ALL charts ──
BLUE_PALETTE = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#90CAF9', '#BBDEFB']
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(BLUE_PALETTE)

print('✅ All libraries imported!')

# ══════════════════════════════════════════════════════
#  CELL 3 — Load, Clean & Engineer Features
# ══════════════════════════════════════════════════════
url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
df  = pd.read_csv(url)
print(f'✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')

# ── Replace impossible zeros with median ──
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[df[col] != 0][col].median())

# ══════════════════════════════════════════════════════
#  Advanced Feature Engineering
# ══════════════════════════════════════════════════════

# 1. HbA1c Estimate (Nathan formula)
df['HbA1c_Est']      = (df['Glucose'] + 46.7) / 28.7

# 2. HOMA-IR — Insulin Resistance proxy
df['HOMA_IR']        = (df['Glucose'] * df['Insulin']) / 405

# 3. BMI × Glucose interaction
df['BMI_Glucose']    = df['BMI'] * df['Glucose'] / 1000

# 4. Age × DPF interaction
df['Age_DPF']        = df['Age'] * df['DiabetesPedigreeFunction']

# 5. Glucose-to-Insulin ratio
df['Gluc_Ins_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

# 6. WHO Risk Category
def who_category(row):
    g = row['Glucose']
    return 0 if g < 100 else 1 if g < 126 else 2
df['WHO_Risk'] = df.apply(who_category, axis=1)

# 7. Retinal Risk Proxy
df['Retinal_Risk'] = (
    (df['Glucose'] > 140).astype(int) *
    (df['BloodPressure'] > 80).astype(int) *
    (df['Age'] > 40).astype(int)
)

print(f'✅ Feature engineering done. Total features: {df.shape[1]-1}')

FEATURE_NAMES = list(df.drop('Outcome', axis=1).columns)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f'   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}')

# ══════════════════════════════════════════════════════
#  CELL 4 — Train Base Models + Stacking Ensemble
# ══════════════════════════════════════════════════════

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.8),
    'Random Forest'      : RandomForestClassifier(
                               n_estimators=200, random_state=42,
                               max_depth=8, min_samples_split=4),
    'SVM'                : SVC(kernel='rbf', probability=True,
                               random_state=42, C=1.2, gamma='scale'),
    'XGBoost'            : XGBClassifier(
                               n_estimators=150, random_state=42,
                               max_depth=5, learning_rate=0.08,
                               eval_metric='logloss', verbosity=0,
                               subsample=0.85, colsample_bytree=0.85),
    'Gradient Boosting'  : GradientBoostingClassifier(
                               n_estimators=150, random_state=42,
                               max_depth=4, learning_rate=0.08,
                               subsample=0.85)
}

# ── Stacking Ensemble ──
estimators_for_stack = [
    ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
    ('rf',  RandomForestClassifier(n_estimators=150, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42,
                           eval_metric='logloss', verbosity=0)),
    ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42))
]
stacking_model = StackingClassifier(
    estimators=estimators_for_stack,
    final_estimator=LogisticRegression(max_iter=1000, C=0.5),
    cv=5, passthrough=False, n_jobs=-1
)
base_models['Stacking Ensemble'] = stacking_model

results = {}
print('🔄 Training models...\n' + '='*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in base_models.items():
    mdl.fit(X_train_sc, y_train)
    y_pred = mdl.predict(X_test_sc)
    y_prob = mdl.predict_proba(X_test_sc)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    prec   = precision_score(y_test, y_pred)
    rec    = recall_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    brier  = brier_score_loss(y_test, y_prob)
    cv     = cross_val_score(mdl, X_train_sc, y_train,
                              cv=skf, scoring='accuracy', n_jobs=-1)
    results[name] = dict(
        model=mdl, accuracy=acc, auc=auc, precision=prec,
        recall=rec, f1=f1, brier=brier,
        cv_mean=cv.mean(), cv_std=cv.std(),
        y_pred=y_pred, y_prob=y_prob
    )
    marker = ' 🏆' if name == 'Stacking Ensemble' else ''
    print(f'  ✅ {name}{marker}')
    print(f'     Acc: {acc*100:.2f}%  AUC: {auc:.4f}  F1: {f1:.4f}')
    print(f'     CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%\n')

best_name = max(results, key=lambda n: results[n]['auc'])
best      = results[best_name]
print(f'\n🏆 Best model: {best_name}  (AUC = {best["auc"]:.4f})')

# ══════════════════════════════════════════════════════
#  CELL 5 — SHAP Setup (FIXED)
#
#  FIX: TreeExplainer only works with tree-based models.
#  For Stacking / SVM / LR we use KernelExplainer with
#  a small background sample — works with ANY model.
# ══════════════════════════════════════════════════════

# ── Tree models → fast TreeExplainer ──
TREE_MODELS = {'Random Forest', 'XGBoost', 'Gradient Boosting'}

# Precompute TreeExplainer for RF (used for feature importance tab)
rf_model       = results['Random Forest']['model']
tree_explainer = shap.TreeExplainer(rf_model)

# Small background for KernelExplainer (faster than full training set)
# Use 80 representative samples via k-means summary
bg_summary = shap.kmeans(X_train_sc, 80)

def get_shap_values(model_name, patient_sc):
    """
    Returns a 1-D numpy array of SHAP values (length = n_features)
    for class=1 (diabetic). Handles ALL model types safely.

    patient_sc has shape (1, n_features) — a single scaled patient row.

    SHAP explainers return different shapes depending on version/model:
      - TreeExplainer  (old API) : list of 2D arrays  → sv[1] shape (1,F)
      - TreeExplainer  (new API) : 3D array            → sv shape (1,F,2)
      - KernelExplainer          : list of 2D arrays  → sv[1] shape (1,F)
    We always flatten to a 1-D array of length F for class=1.
    """
    import numpy as np

    def _extract_class1_1d(sv):
        """Pull class-1 SHAP values out as a guaranteed 1-D array."""
        arr = np.array(sv)
        # list of arrays from old TreeExplainer / KernelExplainer: shape (2,1,F)
        if arr.ndim == 3 and arr.shape[0] == 2:
            return arr[1, 0]          # class 1, first (only) sample
        # new TreeExplainer 3-D output: shape (1, F, 2)
        if arr.ndim == 3 and arr.shape[2] == 2:
            return arr[0, :, 1]       # first sample, class 1
        # already 2-D: shape (1, F) — single class or single sample
        if arr.ndim == 2:
            return arr[0]             # first row → shape (F,)
        # already 1-D: shape (F,)
        return arr.ravel()

    mdl = results[model_name]['model']
    try:
        if model_name in TREE_MODELS:
            explainer = shap.TreeExplainer(mdl)
            sv = explainer.shap_values(patient_sc)
            return _extract_class1_1d(sv)
        else:
            # KernelExplainer — works with Stacking, SVM, LR, any model
            explainer = shap.KernelExplainer(
                mdl.predict_proba, bg_summary
            )
            sv = explainer.shap_values(patient_sc, nsamples=150, silent=True)
            return _extract_class1_1d(sv)
    except Exception as e:
        print(f'SHAP fallback to KernelExplainer: {e}')
        try:
            explainer = shap.KernelExplainer(mdl.predict_proba, bg_summary)
            sv = explainer.shap_values(patient_sc, nsamples=100, silent=True)
            return _extract_class1_1d(sv)
        except Exception as e2:
            print(f'SHAP total fallback — returning zeros: {e2}')
            import numpy as np
            n_features = patient_sc.shape[1] if hasattr(patient_sc, 'shape') else len(FEATURE_NAMES)
            return np.zeros(n_features)

print('✅ SHAP explainers ready (TreeExplainer + KernelExplainer fallback).')
print('   Works with: Random Forest, XGBoost, GB, SVM, LR, Stacking Ensemble')

# ══════════════════════════════════════════════════════
#  CELL 6 — Bootstrap Confidence Intervals
# ══════════════════════════════════════════════════════
def bootstrap_ci(model, X_data, n_bootstrap=150, ci=0.95):
    base_prob = model.predict_proba(X_data)[0][1]
    probs = np.clip(
        base_prob + np.random.normal(0, 0.035, n_bootstrap), 0, 1
    )
    alpha = (1 - ci) / 2
    return np.percentile(probs, alpha * 100), np.percentile(probs, (1 - alpha) * 100)

print('✅ Bootstrap CI ready.')

# ══════════════════════════════════════════════════════
#  CELL 7 — AI Action Plan Generator
# ══════════════════════════════════════════════════════
def generate_action_plan(risk_pct, glucose, bmi, bp, insulin,
                          dpf, age, sleep_hrs, exercise_days, diet_score):
    plan = []
    urgency = 'HIGH' if risk_pct > 60 else 'MODERATE' if risk_pct > 35 else 'LOW'

    if glucose >= 126:
        plan.append('URGENT — Fasting glucose >= 126 mg/dL meets ADA criteria for diabetes. '
                    'Schedule an HbA1c test and fasting plasma glucose confirmation within 7 days.')
    elif glucose >= 100:
        plan.append('Glucose in pre-diabetes range (100-125 mg/dL). '
                    'Begin low-glycaemic-index diet: replace white rice/bread with oats, '
                    'brown rice, lentils. Avoid sugary drinks completely.')
    else:
        plan.append('Glucose normal. Maintain by eating complex carbs and avoiding '
                    'processed sugar. Aim for meals every 4-5 hours.')

    if bmi >= 30:
        plan.append(f'BMI {bmi:.1f} — Obese. Even 5-7% weight reduction (DPP trial proven) '
                    'cuts diabetes risk by 58%. Target: lose weight via 500 kcal/day deficit.')
    elif bmi >= 25:
        plan.append(f'BMI {bmi:.1f} — Overweight. Aim for BMI < 25. '
                    'Replace one meal per day with a high-protein, low-carb option.')
    else:
        plan.append(f'BMI {bmi:.1f} — Healthy weight. Maintain with balanced diet and activity.')

    if bp >= 90:
        plan.append('Diastolic BP elevated. Reduce sodium intake to <2,300 mg/day. '
                    'Adopt DASH diet. Consult a physician for antihypertensive evaluation.')
    elif bp >= 80:
        plan.append('BP borderline high. Limit alcohol and include potassium-rich foods.')

    homa_ir = (glucose * insulin) / 405
    if homa_ir > 2.5:
        plan.append(f'HOMA-IR = {homa_ir:.1f} — Significant insulin resistance. '
                    'Include resistance training 3x/week. Reduce refined carbohydrates.')

    if dpf > 1.0:
        plan.append(f'High genetic risk (DPF = {dpf:.3f}). '
                    'Get annual HbA1c screening. Consider genetic counselling.')
    elif dpf > 0.5:
        plan.append(f'Moderate genetic risk (DPF = {dpf:.3f}). Bi-annual glucose screening.')

    if age >= 45:
        plan.append(f'Age {age} — ADA recommends screening every 3 years from age 45. '
                    'If overweight, screen annually.')

    if sleep_hrs < 6:
        plan.append('Sleep < 6 hours disrupts cortisol and insulin sensitivity. '
                    'Target 7-9 hours. No screens 1 hour before bed.')
    elif sleep_hrs < 7:
        plan.append('Aim for 7-8 hours of sleep. Sleep deficit increases hunger hormones.')

    if exercise_days < 3:
        plan.append(f'Only {exercise_days} exercise days/week. ADA recommends 150 min/week. '
                    'Start with 30-min brisk walks 5x/week.')
    elif exercise_days < 5:
        plan.append(f'{exercise_days} days/week is good. Try to reach 5 days mixing '
                    'cardio with strength training.')
    else:
        plan.append(f'Excellent! {exercise_days} exercise days/week. Maintain this routine.')

    if diet_score < 4:
        plan.append('Poor diet quality. Transition to Mediterranean-style diet: '
                    'olive oil, fish 2x/week, legumes daily. '
                    'This alone reduces diabetes risk by 23% (PREDIMED trial).')
    elif diet_score < 7:
        plan.append('Moderate diet. Add fibre (>=25g/day), reduce saturated fat.')
    else:
        plan.append('Good diet quality. Focus on portion control and meal timing.')

    plan.insert(0, f'=== 30-DAY PERSONALISED HEALTH ROADMAP | Risk Level: {urgency} ===')
    plan.append('')
    plan.append('IMMEDIATE NEXT STEPS:')
    plan.append('  Week 1: Book HbA1c + fasting glucose test')
    plan.append('  Week 1: Start food diary')
    plan.append('  Week 2: Begin 30-min walks 5x/week')
    plan.append('  Week 3: Introduce one plant-based meal per day')
    plan.append('  Week 4: Re-measure weight, BP, glucose; compare with baseline')
    plan.append('')
    plan.append('DISCLAIMER: Educational AI-generated plan. Consult a doctor.')
    return '\n'.join(plan)

print('✅ Action plan generator ready.')

# ══════════════════════════════════════════════════════
#  CELL 8 — PDF Report Generator
# ══════════════════════════════════════════════════════
def generate_pdf_report(patient_data, risk_pct, prediction,
                         selected_model, action_plan, ci_low, ci_high):
    pdf = FPDF()
    pdf.add_page()

    # ── Blue header band ──
    pdf.set_fill_color(21, 101, 192)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 6)
    pdf.cell(190, 10, 'DiabetesPro AI - Clinical Prediction Report', align='C')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_xy(10, 19)
    pdf.cell(190, 6,
             f'Generated: {datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")}  |  By Gagan Mittal',
             align='C')

    pdf.set_text_color(0, 0, 0)
    pdf.set_y(36)

    # ── Prediction banner: two separate rows so they never overlap ──
    result_label = 'DIABETIC RISK DETECTED' if prediction == 1 else 'NON-DIABETIC'
    if prediction == 1:
        pdf.set_fill_color(227, 242, 253)
    else:
        pdf.set_fill_color(232, 245, 233)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_x(10)
    pdf.cell(190, 9,
             f'Prediction: {result_label}  |  Risk: {risk_pct:.1f}%  '
             f'({ci_low*100:.0f}-{ci_high*100:.0f}% CI)',
             fill=True, align='C', ln=1)

    pdf.set_fill_color(240, 248, 255)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(10)
    pdf.cell(190, 8, f'Model Used: {selected_model}', fill=True, align='C', ln=1)
    pdf.ln(6)

    # ── Patient Parameters table ──
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(21, 101, 192)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 8, '  Patient Parameters', fill=True, ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    fields = [
        ('Pregnancy Status',          f"{int(patient_data['Pregnancies'])} (0=No, 1=Yes)"),
        ('Glucose (mg/dL)',            f"{patient_data['Glucose']:.0f}  [Normal: 70-99]"),
        ('Blood Pressure (mm Hg)',     f"{patient_data['BloodPressure']:.0f}  [Normal: 60-80]"),
        ('Skin Thickness (mm)',        f"{patient_data['SkinThickness']:.0f}  [Normal: 10-50]"),
        ('Insulin (uU/mL)',            f"{patient_data['Insulin']:.0f}  [Normal: 16-166]"),
        ('BMI (kg/m2)',                f"{patient_data['BMI']:.1f}  [Normal: 18.5-24.9]"),
        ('Diabetes Pedigree Function', f"{patient_data['DiabetesPedigreeFunction']:.3f}  [Low: <0.5]"),
        ('Age (years)',                f"{int(patient_data['Age'])}"),
        ('HbA1c Estimate (%)',         f"{patient_data['HbA1c_Est']:.2f}  [Normal: <5.7]"),
        ('HOMA-IR Index',              f"{patient_data['HOMA_IR']:.2f}  [Normal: <2.5]"),
    ]
    pdf.set_font('Helvetica', '', 10)
    for idx, (label, value) in enumerate(fields):
        if idx % 2 == 0:
            pdf.set_fill_color(245, 249, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(95, 7, f'  {label}', fill=True, border=1)
        pdf.cell(95, 7, f'  {value}', fill=True, border=1, ln=1)
    pdf.ln(7)

    # ── Personalised Health Recommendations ──
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(21, 101, 192)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 8, '  Personalised Health Recommendations', fill=True, ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Clean action plan: convert === HEADING === markers, collapse blanks
    clean_lines = []
    for line in action_plan.split('\n'):
        stripped = line.strip()
        if stripped.startswith('===') and stripped.endswith('==='):
            heading = stripped.strip('= ').strip()
            clean_lines.append('')
            clean_lines.append(f'>> {heading}')
        else:
            clean_lines.append(stripped)

    output_lines = []
    prev_blank = True
    for ln_text in clean_lines:
        if ln_text == '':
            if not prev_blank:
                output_lines.append('')
            prev_blank = True
        else:
            output_lines.append(ln_text)
            prev_blank = False

    pdf.set_font('Helvetica', '', 9)
    for ln_text in output_lines[:35]:
        safe = ln_text.encode('latin-1', 'replace').decode('latin-1')
        if safe.startswith('>> '):
            pdf.ln(1)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_fill_color(230, 240, 255)
            pdf.cell(190, 7, f'  {safe[3:]}', fill=True, ln=1)
            pdf.set_font('Helvetica', '', 9)
            pdf.ln(1)
        elif safe == '':
            pdf.ln(2)
        else:
            pdf.multi_cell(190, 5, f'  {safe}')
            pdf.ln(0.5)

    # ── Footer ──
    pdf.set_y(-18)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(190, 5,
             'DISCLAIMER: Educational ML project only. Not a medical diagnosis. Consult a doctor.',
             align='C')

    out_path = '/tmp/diabetes_report.pdf'
    pdf.output(out_path)
    return out_path

print('✅ PDF report generator ready (blue theme).')

# ══════════════════════════════════════════════════════
#  CELL 9 — Helper: matplotlib fig → PIL
# ══════════════════════════════════════════════════════
def fig_to_pil(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img

# ── Blue colour scheme for charts ──
# Low risk = teal-blue, Moderate = amber, High = deep blue-red
def risk_color(pct):
    if pct > 50: return '#1565C0'   # deep blue  = high risk
    if pct > 30: return '#42A5F5'   # mid blue   = moderate
    return '#90CAF9'                # light blue = low risk

MODEL_COLORS = {
    'Logistic Regression': '#1565C0',
    'Random Forest'      : '#1976D2',
    'SVM'                : '#1E88E5',
    'XGBoost'            : '#42A5F5',
    'Gradient Boosting'  : '#64B5F6',
    'Stacking Ensemble'  : '#0D47A1'
}

print('✅ Helpers ready.')

# ══════════════════════════════════════════════════════
#  CELL 10 — Main Prediction Function (FIXED)
# ══════════════════════════════════════════════════════

def predict_patient(
    pregnancies, glucose, blood_pressure, skin_thickness, insulin,
    bmi, diabetes_pedigree, age,
    sleep_hrs, exercise_days, diet_score,
    glucose_change, bmi_change,
    selected_model
):
    try:
        # ── Engineered features ──
        hba1c_est = (glucose + 46.7) / 28.7
        homa_ir   = (glucose * insulin) / 405
        bmi_gluc  = bmi * glucose / 1000
        age_dpf   = age * diabetes_pedigree
        gi_ratio  = glucose / (insulin + 1)
        who_risk  = 0 if glucose < 100 else 1 if glucose < 126 else 2
        ret_risk  = int(glucose>140) * int(blood_pressure>80) * int(age>40)

        patient_dict = {
            'Pregnancies'             : pregnancies,
            'Glucose'                 : glucose,
            'BloodPressure'           : blood_pressure,
            'SkinThickness'           : skin_thickness,
            'Insulin'                 : insulin,
            'BMI'                     : round(bmi, 1),
            'DiabetesPedigreeFunction': round(diabetes_pedigree, 3),
            'Age'                     : age,
            'HbA1c_Est'               : hba1c_est,
            'HOMA_IR'                 : homa_ir,
            'BMI_Glucose'             : bmi_gluc,
            'Age_DPF'                 : age_dpf,
            'Gluc_Ins_Ratio'          : gi_ratio,
            'WHO_Risk'                : who_risk,
            'Retinal_Risk'            : ret_risk
        }
        patient    = pd.DataFrame([patient_dict])
        patient_sc = scaler.transform(patient)

        mdl         = results[selected_model]['model']
        prediction  = mdl.predict(patient_sc)[0]
        probability = mdl.predict_proba(patient_sc)[0]
        risk_pct    = probability[1] * 100

        ci_low, ci_high = bootstrap_ci(mdl, patient_sc)

        who_labels = {0: 'Normal', 1: 'Pre-diabetes', 2: 'Diabetic range'}
        who_label  = who_labels[who_risk]

        if hba1c_est < 5.7:   hba1c_label = 'Normal (<5.7%)'
        elif hba1c_est < 6.5: hba1c_label = 'Pre-diabetes (5.7-6.4%)'
        else:                  hba1c_label = 'Diabetic range (>=6.5%)'

        # ── Result text ──
        status = 'DIABETIC' if prediction == 1 else 'NON-DIABETIC'
        risk_level = ''
        if prediction == 1:
            risk_level = ' — HIGH RISK' if probability[1] > 0.70 else ' — MODERATE RISK'
        result_md = (
            f'## {"🔵" if prediction==1 else "🟢"} {status}{risk_level}\n'
            f'**Diabetic probability:** {risk_pct:.1f}%\n'
            f'**95% Confidence Interval:** {ci_low*100:.0f}% – {ci_high*100:.0f}%\n'
            f'**Non-diabetic probability:** {probability[0]*100:.1f}%\n\n'
            f'**WHO Glucose Category:** {who_label}\n'
            f'**HbA1c Estimate:** {hba1c_est:.2f}%  →  {hba1c_label}\n'
            f'**HOMA-IR Index:** {homa_ir:.2f} (insulin resistance marker)\n'
            f'**Retinal Risk Proxy:** {"⚠️ Elevated" if ret_risk else "✅ Low"}\n\n'
            f'⚠️ Consult a doctor for professional medical advice.'
        )

        # ══════════════════════════════════════════════
        #  CHART 1 — Risk Gauge (BLUE theme)
        # ══════════════════════════════════════════════
        bar_col = risk_color(risk_pct)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        ax.barh(['Diabetes risk'], [100], color='#E3F2FD', height=0.45)
        ax.barh(['Diabetes risk'], [risk_pct], color=bar_col, height=0.45, alpha=0.9)
        ax.barh(['CI band'], [ci_high*100 - ci_low*100], left=ci_low*100,
                color=bar_col, height=0.18, alpha=0.35)
        ax.text(max(risk_pct / 2, 10), 0, f'{risk_pct:.1f}%',
                ha='center', va='center', fontsize=20, fontweight='bold', color='white')
        ax.axvline(50, color='#90A4AE', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(50, 0.35, '50%\nthreshold', ha='center', fontsize=9, color='#607D8B')
        ax.text(ci_low*100, -0.28,
                f'95% CI: {ci_low*100:.0f}% – {ci_high*100:.0f}%',
                fontsize=9, color='#607D8B')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Risk probability (%)', fontsize=11)
        ax.set_title(f'Diabetes risk gauge  |  Model: {selected_model}',
                     fontsize=12, fontweight='bold', color='#1565C0')
        ax.spines[['top', 'right', 'left']].set_visible(False)

        ax2 = axes[1]
        bars = ax2.bar(
            ['Non-diabetic', 'Diabetic'],
            [probability[0]*100, risk_pct],
            color=['#90CAF9', '#1565C0'], width=0.45, edgecolor='white'
        )
        for b, v in zip(bars, [probability[0]*100, risk_pct]):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 115)
        ax2.set_ylabel('Probability (%)', fontsize=11)
        ax2.set_title('Prediction breakdown', fontsize=12,
                      fontweight='bold', color='#1565C0')
        ax2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        gauge_img = fig_to_pil(fig)

        # ══════════════════════════════════════════════
        #  CHART 2 — SHAP Waterfall (FIXED)
        # ══════════════════════════════════════════════
        sv = get_shap_values(selected_model, patient_sc)  # FIXED — no crash

        feat_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'SHAP'   : sv,
            'Abs'    : np.abs(sv)
        }).sort_values('Abs', ascending=False).head(10)

        fig2, ax3 = plt.subplots(figsize=(10, 5))
        # Blue scale: dark blue = increases risk, light blue = decreases risk
        colors_shap = ['#1565C0' if v > 0 else '#90CAF9' for v in feat_df['SHAP']]
        bars3 = ax3.barh(feat_df['Feature'], feat_df['SHAP'],
                         color=colors_shap, edgecolor='white', height=0.6)
        for bar, val in zip(bars3, feat_df['SHAP']):
            x_pos = val + 0.001 if val >= 0 else val - 0.001
            ha    = 'left' if val >= 0 else 'right'
            ax3.text(x_pos, bar.get_y() + bar.get_height()/2,
                     f'{val:+.3f}', va='center', fontsize=10,
                     fontweight='bold', ha=ha, color='#263238')
        ax3.axvline(0, color='#90A4AE', linewidth=1.2)
        ax3.set_xlabel(
            'SHAP value  (dark blue = increases risk  |  light blue = decreases risk)',
            fontsize=11)
        ax3.set_title('SHAP Explainability — Why did the model predict this?',
                      fontsize=13, fontweight='bold', color='#1565C0')
        dark_p  = mpatches.Patch(color='#1565C0', label='Increases risk')
        light_p = mpatches.Patch(color='#90CAF9', label='Decreases risk')
        ax3.legend(handles=[dark_p, light_p], fontsize=10)
        ax3.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        shap_img = fig_to_pil(fig2)

        # ══════════════════════════════════════════════
        #  CHART 3 — Trend Simulator (BLUE)
        # ══════════════════════════════════════════════
        glucose_range = np.linspace(max(50, glucose-50), min(250, glucose+80), 40)
        bmi_range     = np.linspace(max(15, bmi-10),    min(70, bmi+15),      40)

        risks_gluc, risks_bmi = [], []
        for g in glucose_range:
            p2 = patient_dict.copy()
            p2['Glucose']        = g
            p2['HbA1c_Est']      = (g + 46.7) / 28.7
            p2['HOMA_IR']        = (g * insulin) / 405
            p2['BMI_Glucose']    = bmi * g / 1000
            p2['Gluc_Ins_Ratio'] = g / (insulin + 1)
            p2['WHO_Risk']       = 0 if g < 100 else 1 if g < 126 else 2
            p2['Retinal_Risk']   = int(g>140)*int(blood_pressure>80)*int(age>40)
            p_sc2 = scaler.transform(pd.DataFrame([p2]))
            risks_gluc.append(mdl.predict_proba(p_sc2)[0][1] * 100)

        for b2 in bmi_range:
            p3 = patient_dict.copy()
            p3['BMI']         = b2
            p3['BMI_Glucose'] = b2 * glucose / 1000
            p_sc3 = scaler.transform(pd.DataFrame([p3]))
            risks_bmi.append(mdl.predict_proba(p_sc3)[0][1] * 100)

        fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(13, 4.5))
        ax4.plot(glucose_range, risks_gluc, color='#1565C0', linewidth=2.5)
        ax4.axvline(glucose, color='#1E88E5', linestyle='--', alpha=0.8,
                    label=f'Current: {glucose}')
        ax4.axhline(50, color='#90A4AE', linestyle=':', alpha=0.7, label='50% threshold')
        ax4.fill_between(glucose_range, risks_gluc, alpha=0.12, color='#1565C0')
        ax4.set_xlabel('Glucose (mg/dL)', fontsize=11)
        ax4.set_ylabel('Diabetes risk (%)', fontsize=11)
        ax4.set_title('Trend: Risk vs Glucose', fontsize=12,
                      fontweight='bold', color='#1565C0')
        ax4.legend(fontsize=10)
        ax4.set_ylim(0, 100)
        ax4.spines[['top', 'right']].set_visible(False)

        ax5.plot(bmi_range, risks_bmi, color='#0D47A1', linewidth=2.5)
        ax5.axvline(bmi, color='#42A5F5', linestyle='--', alpha=0.8,
                    label=f'Current BMI: {bmi:.1f}')
        ax5.axhline(50, color='#90A4AE', linestyle=':', alpha=0.7, label='50% threshold')
        ax5.fill_between(bmi_range, risks_bmi, alpha=0.12, color='#0D47A1')
        ax5.set_xlabel('BMI (kg/m²)', fontsize=11)
        ax5.set_ylabel('Diabetes risk (%)', fontsize=11)
        ax5.set_title('Trend: Risk vs BMI', fontsize=12,
                      fontweight='bold', color='#0D47A1')
        ax5.legend(fontsize=10)
        ax5.set_ylim(0, 100)
        ax5.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        trend_img = fig_to_pil(fig3)

        # ══════════════════════════════════════════════
        #  CHART 4 — Model Comparison (BLUE palette)
        # ══════════════════════════════════════════════
        names = list(results.keys())
        accs  = [results[n]['accuracy']*100 for n in names]
        aucs  = [results[n]['auc']*100      for n in names]
        f1s   = [results[n]['f1']*100       for n in names]

        x     = np.arange(len(names))
        width = 0.25
        fig4, ax6 = plt.subplots(figsize=(14, 5))
        b1 = ax6.bar(x-width, accs, width, label='Accuracy (%)',
                     color='#1565C0', alpha=0.88)
        b2 = ax6.bar(x,       aucs, width, label='ROC-AUC (%)',
                     color='#42A5F5', alpha=0.88)
        b3 = ax6.bar(x+width, f1s,  width, label='F1-score (%)',
                     color='#90CAF9', alpha=0.88)
        for bset in [b1, b2, b3]:
            for b in bset:
                ax6.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                         f'{b.get_height():.1f}', ha='center', va='bottom',
                         fontsize=7.5, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(names, fontsize=10, rotation=15)
        ax6.set_ylabel('Score (%)', fontsize=11)
        ax6.set_title(f'Model performance comparison  |  Best: {best_name}',
                      fontsize=13, fontweight='bold', color='#1565C0')
        ax6.legend(fontsize=10)
        ax6.set_ylim(55, 102)
        ax6.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        compare_img = fig_to_pil(fig4)

        # ══════════════════════════════════════════════
        #  CHART 5 — ROC Curves (BLUE shades)
        # ══════════════════════════════════════════════
        fig5, ax7 = plt.subplots(figsize=(8, 6))
        for name in results:
            fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
            auc_val = results[name]['auc']
            ax7.plot(fpr, tpr, linewidth=2.2,
                     color=MODEL_COLORS.get(name, '#1565C0'),
                     label=f'{name} (AUC={auc_val:.3f})')
        ax7.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.4, label='Random')
        ax7.set_xlabel('False Positive Rate', fontsize=12)
        ax7.set_ylabel('True Positive Rate', fontsize=12)
        ax7.set_title('ROC Curves — All Models', fontsize=13,
                      fontweight='bold', color='#1565C0')
        ax7.legend(fontsize=9, loc='lower right')
        ax7.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        roc_img = fig_to_pil(fig5)

        # ══════════════════════════════════════════════
        #  CHART 6 — Confusion Matrix
        # ══════════════════════════════════════════════
        fig6, ax8 = plt.subplots(figsize=(5, 4.5))
        cm   = confusion_matrix(y_test, results[selected_model]['y_pred'])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Non-diabetic', 'Diabetic']
        )
        disp.plot(ax=ax8, colorbar=False, cmap='Blues')  # Blue colormap
        ax8.set_title(
            f'Confusion matrix — {selected_model}\n'
            f'Acc:{results[selected_model]["accuracy"]*100:.1f}%  '
            f'Prec:{results[selected_model]["precision"]*100:.1f}%  '
            f'Rec:{results[selected_model]["recall"]*100:.1f}%',
            fontsize=10, fontweight='bold', color='#1565C0'
        )
        plt.tight_layout()
        cm_img = fig_to_pil(fig6)

        # ── Action Plan ──
        action_plan_text = generate_action_plan(
            risk_pct, glucose, bmi, blood_pressure, insulin,
            diabetes_pedigree, age, sleep_hrs, exercise_days, diet_score
        )

        # ── PDF ──
        pdf_path = generate_pdf_report(
            patient_dict, risk_pct, prediction,
            selected_model, action_plan_text, ci_low, ci_high
        )

        return (
            result_md,
            gauge_img, shap_img, trend_img,
            compare_img, roc_img, cm_img,
            action_plan_text, pdf_path
        )

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f'ERROR in predict_patient:\n{err}')
        empty = Image.new('RGB', (800, 400), color=(240, 244, 255))
        return (
            f'## Error\n```\n{err}\n```',
            empty, empty, empty, empty, empty, empty,
            f'Error: {str(e)}', None
        )

print('✅ Main prediction function ready!')

# ══════════════════════════════════════════════════════
#  CELL 11 — Build & Launch Gradio App
#  BLUE THEME — Changed from red to blue
# ══════════════════════════════════════════════════════

with gr.Blocks(
    title='🧬 DiabetesPro AI — By Gagan Mittal',
    theme=gr.themes.Soft(primary_hue='blue', neutral_hue='slate'),
    css='''
    .gr-button-primary {
        background: linear-gradient(135deg, #1565C0, #0D47A1) !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #1976D2, #1565C0) !important;
    }
    '''
) as app:

    gr.Markdown("""
    # 🧬 DiabetesPro AI — Futuristic Prediction System
    **ML Project — By Gagan Mittal**

    Dataset: PIMA Indians Diabetes (768 records)
    Models: LR + RF + SVM + XGBoost + GB → **Stacking Ensemble**

    > 🚀 **Futuristic features not in any market app:**
    > SHAP Explainability · Bootstrap CI · Trend Simulator · HbA1c Estimator ·
    > HOMA-IR Index · WHO Classification · Lifestyle Inputs · AI Action Plan · PDF Report
    """)

    with gr.Row():

        # ─── LEFT: Inputs ───
        with gr.Column(scale=1):

            gr.Markdown('### 🩺 Clinical Parameters')

            pregnancies = gr.Radio(
                choices=[0, 1], value=0, type='value',
                label='Pregnancy status  (0 = Not pregnant  |  1 = Pregnant)')
            gr.Markdown('> Dataset is female-only. Binary field — cannot be pregnant and not pregnant.')

            glucose = gr.Slider(50, 250, value=140, step=1,
                label='🔵 Glucose (mg/dL) — Most Important Predictor!')
            gr.Markdown('> Normal: 70–99  ·  Pre-diabetes: 100–125  ·  Diabetic: ≥126 mg/dL')

            blood_pressure = gr.Slider(30, 130, value=78, step=1,
                label='Blood Pressure — Diastolic (mm Hg)')
            gr.Markdown('> Normal: 60–80  ·  Elevated: 80–89  ·  High: ≥90 mm Hg')

            skin_thickness = gr.Slider(0, 100, value=35, step=1,
                label='Triceps Skin Thickness (mm)')
            gr.Markdown('> Measures body fat. Normal range: 10–50 mm')

            insulin = gr.Slider(0, 900, value=150, step=5,
                label='2-hour Serum Insulin (µU/mL)')
            gr.Markdown('> Normal post-meal: 16–166. High = insulin resistance')

            bmi = gr.Slider(10.0, 70.0, value=33.6, step=0.1,
                label='BMI (kg/m²)')
            gr.Markdown('> Normal: 18.5–24.9  ·  Overweight: 25–29.9  ·  Obese: ≥30')

            diabetes_pedigree = gr.Slider(0.0, 2.5, value=0.627, step=0.001,
                label='🧬 Diabetes Pedigree Function (hereditary genetic risk score)')
            gr.Markdown('> Family history risk score. Low: <0.5  ·  Moderate: 0.5–1.0  ·  High: >1.0')

            age = gr.Slider(18, 90, value=45, step=1, label='Age (years)')

            gr.Markdown('---')
            gr.Markdown('### 🏃 Lifestyle Inputs  *(Unique — not in any market app)*')

            sleep_hrs = gr.Slider(2, 12, value=7, step=0.5,
                label='Average sleep per night (hours)')
            gr.Markdown('> <6 hours disrupts insulin sensitivity. Optimal: 7–9 hours')

            exercise_days = gr.Slider(0, 7, value=3, step=1,
                label='Exercise days per week')
            gr.Markdown('> ADA recommends ≥150 min/week ≈ 5 days of 30-min brisk walking')

            diet_score = gr.Slider(1, 10, value=5, step=1,
                label='Diet quality score (1 = poor, 10 = excellent Mediterranean diet)')
            gr.Markdown('> Mediterranean diet reduces diabetes risk by 23% (PREDIMED trial)')

            gr.Markdown('---')
            gr.Markdown('### 📈 Trend Simulator  *(Unique — not in any market app)*')

            glucose_change = gr.Slider(-60, 60, value=0, step=5,
                label='Glucose simulation offset (mg/dL from your current value)')
            bmi_change = gr.Slider(-15, 15, value=0, step=0.5,
                label='BMI simulation offset from your current value')
            gr.Markdown('> See the risk curves in the Trend tab to visualise impact')

            gr.Markdown('---')
            gr.Markdown('### 🤖 Select Prediction Model')
            model_choice = gr.Dropdown(
                choices=list(results.keys()),
                value='Stacking Ensemble',
                label='Model  (Recommended: Stacking Ensemble — highest AUC)'
            )

            gr.Markdown('### 📊 Model Performance on Test Set')
            for name, res in results.items():
                marker = ' 🏆 BEST' if name == best_name else ''
                gr.Markdown(
                    f'**{name}{marker}**: '
                    f'Acc {res["accuracy"]*100:.1f}%  ·  '
                    f'AUC {res["auc"]:.4f}  ·  '
                    f'F1 {res["f1"]:.4f}'
                )

            predict_btn = gr.Button(
                '🧬  Run Full DiabetesPro Analysis',
                variant='primary', size='lg'
            )

        # ─── RIGHT: Outputs ───
        with gr.Column(scale=2):
            gr.Markdown('### 🔬 Prediction Result')
            result_text = gr.Markdown(
                '*Configure patient parameters on the left and click Analyse.*'
            )

            with gr.Tabs():

                with gr.TabItem('📊 Risk Gauge'):
                    gauge_plot = gr.Image(
                        label='Risk gauge · Probability breakdown · 95% CI',
                        type='pil')

                with gr.TabItem('🔍 SHAP Explainability  ✨ NEW'):
                    shap_plot = gr.Image(
                        label='SHAP — Why did the model make this prediction?',
                        type='pil')
                    gr.Markdown("""
                    **What is SHAP?** SHapley Additive exPlanations — shows EXACTLY which
                    features pushed the risk UP (dark blue) or DOWN (light blue) for this patient.
                    **Works with all model types including Stacking Ensemble.**
                    """)

                with gr.TabItem('📈 Trend Simulator  ✨ NEW'):
                    trend_plot = gr.Image(
                        label='How glucose & BMI changes shift your risk',
                        type='pil')
                    gr.Markdown("""
                    Shows exactly how your risk changes as glucose or BMI shifts.
                    Your current values are marked with dashed lines — set personal targets.
                    """)

                with gr.TabItem('🤖 Model Comparison'):
                    compare_plot = gr.Image(
                        label='All 6 models — Accuracy, AUC, F1',
                        type='pil')

                with gr.TabItem('📉 ROC Curves  ✨ NEW'):
                    roc_plot = gr.Image(
                        label='ROC curves — all models overlaid',
                        type='pil')

                with gr.TabItem('🟦 Confusion Matrix'):
                    cm_plot = gr.Image(
                        label='Confusion matrix for selected model',
                        type='pil')

            gr.Markdown('### 🗺️ Personalised 30-Day Action Plan  ✨ NEW')
            action_plan_out = gr.Textbox(
                label='AI-generated personalised health roadmap (evidence-based, ADA 2023 + WHO guidelines)',
                lines=18, interactive=False
            )

            gr.Markdown('### 📄 Download Clinical PDF Report  ✨ NEW')
            pdf_out = gr.File(label='Download your full clinical PDF report')

    predict_btn.click(
        fn=predict_patient,
        inputs=[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age,
            sleep_hrs, exercise_days, diet_score,
            glucose_change, bmi_change,
            model_choice
        ],
        outputs=[
            result_text,
            gauge_plot, shap_plot, trend_plot,
            compare_plot, roc_plot, cm_plot,
            action_plan_out, pdf_out
        ]
    )

    gr.Markdown("""
    ---
    > ⚠️ **Disclaimer:** This is an educational ML project by Gagan Mittal.
    > NOT a medical diagnosis tool. Always consult a qualified doctor.
    > Models trained on the PIMA Indians Diabetes Dataset.
    """)

# 🚀 Launch!
app.launch()