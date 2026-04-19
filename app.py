import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")


# =========================
# Clinical Category Logic
# =========================

def glucose_category(glucose):

    if glucose < 70:
        return "LOW"

    elif 70 <= glucose <= 99:
        return "NORMAL"

    elif 100 <= glucose <= 125:
        return "PRE-DIABETES"

    else:
        return "DIABETIC RANGE"


def bmi_category(bmi):

    if bmi < 18.5:
        return "UNDERWEIGHT"

    elif bmi < 25:
        return "NORMAL"

    elif bmi < 30:
        return "OVERWEIGHT"

    else:
        return "OBESE"


def bp_category(bp):

    if bp < 60:
        return "LOW"

    elif bp <= 80:
        return "NORMAL"

    elif bp <= 89:
        return "ELEVATED"

    else:
        return "HIGH"


# =========================
# Probability Chart
# =========================

def create_probability_chart(probability):

    diabetic = probability * 100
    non_diabetic = 100 - diabetic

    labels = ["Non-diabetic", "Diabetic"]
    values = [non_diabetic, diabetic]

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values)

    ax.set_ylabel("Probability (%)")
    ax.set_title("Prediction Breakdown")

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center"
        )

    ax.set_ylim(0, 100)

    return fig


# =========================
# PDF Report Generator
# =========================

def generate_pdf(result, probability, glucose, bp, bmi, homa):

    pdf_file = "diabetes_report.pdf"

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(pdf_file)

    content = []

    title = Paragraph(
        "Diabetes Clinical Prediction Report",
        styles["Title"]
    )

    content.append(title)

    content.append(Spacer(1, 20))

    summary = Paragraph(
        f"Prediction: {result}",
        styles["BodyText"]
    )

    prob_text = Paragraph(
        f"Diabetic probability: {round(probability*100,1)}%",
        styles["BodyText"]
    )

    content.append(summary)
    content.append(prob_text)

    content.append(Spacer(1, 20))

    table_data = [

        ["Glucose", f"{glucose} → {glucose_category(glucose)}"],

        ["Blood Pressure", f"{bp} → {bp_category(bp)}"],

        ["BMI", f"{bmi} → {bmi_category(bmi)}"],

        ["HOMA-IR", str(homa)]

    ]

    table = Table(table_data)

    content.append(table)

    doc.build(content)

    return pdf_file


# =========================
# Prediction Function
# =========================

def predict_diabetes(

        pregnancy_status,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age

):

    try:

        preg = 0 if pregnancy_status == "Not Pregnant" else 1

        input_data = np.array([
            [
                preg,
                glucose,
                bp,
                skin,
                insulin,
                bmi,
                dpf,
                age
            ]
        ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        homa = round(insulin * bmi / 405, 2)

        if probability >= 0.7:
            risk = "HIGH RISK"

        elif probability >= 0.5:
            risk = "MODERATE RISK"

        else:
            risk = "LOW RISK"

        if prediction == 1:
            result = f"🔴 DIABETIC — {risk}"

        else:
            result = f"🟢 NON-DIABETIC — {risk}"

        chart = create_probability_chart(probability)

        clinical_text = f"""
DIABETIC PROBABILITY: {round(probability*100,1)}%

Glucose: {glucose} → {glucose_category(glucose)}
Normal: 70–99 · Pre-diabetes: 100–125 · Diabetic: ≥126 mg/dL

Blood Pressure: {bp} → {bp_category(bp)}
Normal: 60–80 · Elevated: 80–89 · High: ≥90 mm Hg

BMI: {bmi} → {bmi_category(bmi)}
Normal: 18.5–24.9 · Overweight: 25–29.9 · Obese: ≥30

HOMA-IR Index: {homa}

⚠️ Consult a doctor for professional medical advice.
"""

        pdf_file = generate_pdf(

            result,
            probability,
            glucose,
            bp,
            bmi,
            homa

        )

        return result, chart, clinical_text, pdf_file

    except Exception as e:

        return str(e), None, "", None


# =========================
# UI Layout
# =========================

with gr.Blocks(title="DiabetesPro AI") as demo:

    gr.Markdown(
        """
# 🧬 DiabetesPro AI — Clinical Prediction System

This system supports both male and female patients. Pregnancy status applies only to female patients (0 = Not Pregnant, 1 = Pregnant).
"""
    )

    pregnancy = gr.Radio(

        ["Not Pregnant", "Pregnant"],

        value="Not Pregnant",

        label="Pregnancy status (0 = Not pregnant | 1 = Pregnant)"

    )

    glucose = gr.Slider(

        50,
        250,
        value=120,

        label="Glucose (mg/dL)"

    )

    bp = gr.Slider(

        30,
        130,
        value=70,

        label="Blood Pressure — Diastolic (mm Hg)"

    )

    skin = gr.Slider(

        0,
        100,
        value=20,

        label="Triceps Skin Thickness (mm)"

    )

    insulin = gr.Slider(

        0,
        900,
        value=80,

        label="2-hour Serum Insulin (µU/mL)"

    )

    bmi = gr.Slider(

        10,
        70,
        value=30,

        label="BMI (kg/m²)"

    )

    dpf = gr.Slider(

        0,
        2.5,
        value=0.5,

        label="Diabetes Pedigree Function"

    )

    age = gr.Slider(

        18,
        90,
        value=30,

        label="Age (years)"

    )

    submit = gr.Button(

        "Run Full Diabetes Analysis"

    )

    result_box = gr.Textbox(

        label="Prediction Result"

    )

    chart = gr.Plot(

        label="Probability Chart"

    )

    clinical = gr.Textbox(

        label="Clinical Interpretation"

    )

    pdf_file = gr.File(

        label="Download Clinical PDF Report"

    )

    submit.click(

        predict_diabetes,

        inputs=[

            pregnancy,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age

        ],

        outputs=[

            result_box,
            chart,
            clinical,
            pdf_file

        ]

    )

demo.launch()