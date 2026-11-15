# -*- coding: utf-8 -*-
"""
30-day readmission risk portal (diabetes patients, 1999-2008 US hospitals).
Loads a model trained earlier. Does NOT retrain and does NOT need the full dataset.

Files needed in the same folder:
  readmission_diabetes_cache.joblib  (trained model, from the training run)
  app_assets.json                    (choices, defaults, presets, test-set results)
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "readmission_diabetes_cache.joblib"
ASSETS_PATH = "app_assets.json"
DISCLAIMER = "This project is a prototype and is not medical advice."
THRESH_MIN, THRESH_MAX = 0.02, 0.50

NUMERIC_IN_APP = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures",
                  "num_medications", "number_outpatient", "number_emergency",
                  "number_inpatient", "number_diagnoses"]
DRUG_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
             "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
             "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
             "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
             "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
TR_LABELS = {
    "age": "Patient Age (years)",
    "time_in_hospital": "Length of Stay (days)",
    "num_lab_procedures": "Lab Procedures This Visit",
    "num_procedures": "Non-lab Procedures This Visit",
    "num_medications": "Distinct Medications Prescribed",
    "number_outpatient": "Outpatient Visits in Prior Year",
    "number_emergency": "Emergency Visits in Prior Year",
    "number_inpatient": "Inpatient Admissions in Prior Year",
    "number_diagnoses": "Number of Diagnoses",
    "diag_1": "Primary Diagnosis Group",
    "diag_2": "Secondary Diagnosis Group",
    "diag_3": "Additional Diagnosis Group",
    "admission_type_id": "Admission Type",
    "discharge_disposition_id": "Discharge Disposition",
    "admission_source_id": "Admission Source",
    "payer_code": "Payer",
    "medical_specialty": "Admitting Medical Specialty",
    "max_glu_serum": "Max Glucose Serum Test Result",
    "A1Cresult": "HbA1c Test Result",
    "change": "Diabetes Medication Changed",
    "diabetesMed": "Diabetes Medication Prescribed",
    "prior_visits": "Total Prior Visits (computed)",
    "num_drugs_active": "Diabetes Drugs Active (computed)",
}
PRIMARY_COLS = ["age", "time_in_hospital", "number_inpatient", "number_emergency",
                "number_outpatient", "num_medications", "diag_1", "A1Cresult"]

# ---------- Load model and assets ----------
cache = joblib.load(MODEL_PATH)
model = cache["model"]
calibrator = joblib.load("calibrator.joblib")

with open(ASSETS_PATH) as f:
    A = json.load(f)
UI_COLS = A["ui_cols"]
NUM_COLS = A["num_cols"]
MODEL_FEATURES = A["model_features"]
CHOICES = A["choices"]
DEFAULTS = A["defaults"]
PRESETS = A["presets"]
PRESET_TRUTH = A["preset_truth"]
THRESHOLD = float(A["threshold"])          # calibrated threshold from app_assets.json
Y_TEST = np.array(A["test_labels"])
P_TEST = np.array(A["test_probs"])


# ---------- Same preparation and features as training ----------
def prep_common(df):
    df = df.copy()
    for c in df.columns:
        if c in NUMERIC_IN_APP:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].where(df[c].notna(), "Missing").astype(str)
    return df


def add_features(df):
    df = df.copy()
    df["prior_visits"] = df[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1)
    drugs = [c for c in DRUG_COLS if c in df.columns]
    df["num_drugs_active"] = (df[drugs] != "No").sum(axis=1)
    return df


def build_row(values):
    ui = pd.DataFrame([dict(zip(UI_COLS, values))])
    ui = prep_common(ui)
    return add_features(ui).reindex(columns=MODEL_FEATURES)


def summarize(y_true, pred):
    tn, fp, fn, tp = np.bincount(y_true * 2 + pred, minlength=4)[[0, 1, 2, 3]]
    n = len(y_true)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return dict(n=n, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                accuracy=(tp + tn) / n, recall=recall, precision=precision,
                flag_rate=(tp + fp) / n)


def metrics_text(threshold):
    r = summarize(Y_TEST, (P_TEST >= threshold).astype(int))
    return (
        f"Held-out TEST set ({r['n']} encounters) at decision threshold {threshold:.2f}\n"
        f"- FN (Missed Readmission <30d): {r['FN']} | - FP (False Alarms): {r['FP']}\n"
        f"- TP (Correctly Flagged): {r['TP']} | - TN (Correctly Flagged Low): {r['TN']}\n"
        f"- Accuracy: {r['accuracy']:.3f} | Recall: {r['recall']:.3f} | "
        f"Precision: {r['precision']:.3f} | Flagged: {r['flag_rate']:.1%}"
    )


def predict(threshold, *values):
    row = build_row(values)
    prob_raw = float(model.predict_proba(row)[0, 1])
    prob = float(calibrator.predict([prob_raw])[0])
    label = ("HIGH 30-DAY READMISSION RISK WARNING (1)" if prob >= threshold
             else "LOW / MODERATE 30-DAY READMISSION RISK (0)")
    text = f"- {label} - Estimated probability: {prob * 100:.1f}% ({prob:.3f})"
    return text, metrics_text(threshold), DISCLAIMER


def run_preset(which, threshold):
    preset = PRESETS.get(which)
    if preset is None:
        msg = f"No {which} case exists in the test set at this threshold."
        return [gr.update() for _ in UI_COLS] + [msg, metrics_text(threshold), DISCLAIMER]
    vals = [preset[c] for c in UI_COLS]
    text, metrics, notice = predict(threshold, *vals)
    text += f"\n- Verified test-set outcome: {PRESET_TRUTH[which]}"
    return vals + [text, metrics, notice]


# ---------- Interface ----------
def make_component(col):
    label = TR_LABELS.get(col, col.replace("_", " ").title())
    default = DEFAULTS.get(col)
    if col in NUM_COLS:
        return gr.Number(value=default, label=label)
    return gr.Dropdown(choices=CHOICES[col], value=default, label=label)


with gr.Blocks(title="Diabetes Readmission Risk Portal") as demo:
    gr.Markdown("# 30-Day Readmission Risk Portal (Diabetes Patients)")
    gr.Markdown(
        "**Designed for discharge coordinators and nurses** to estimate the chance that a diabetic "
        "patient is readmitted within 30 days. Preset cases come from the held-out test set. "
        "Changing a field does not update the result until you press **Evaluate**."
    )
    gr.Markdown(f"> **Legal Disclaimer:** {DISCLAIMER}")

    with gr.Row():
        btn_fn = gr.Button("Load Case 1: Missed Readmission (False Negative)")
        btn_fp = gr.Button("Load Case 2: False Alarm (False Positive)")

    th_slider = gr.Slider(
        minimum=THRESH_MIN, maximum=THRESH_MAX, step=0.01, value=THRESHOLD,
        label="Decision Threshold (lowering reduces missed readmissions but increases alerts)",
    )

    gr.Markdown("### Primary Clinical Variables")
    primary = [c for c in PRIMARY_COLS if c in UI_COLS]
    secondary = [c for c in UI_COLS if c not in primary]
    comps = {}
    with gr.Row():
        for c in primary[:4]:
            comps[c] = make_component(c)
    with gr.Row():
        for c in primary[4:]:
            comps[c] = make_component(c)
    with gr.Accordion("Optional Demographics & Secondary Indicators", open=False):
        for c in secondary:
            comps[c] = make_component(c)
    ordered = [comps[c] for c in UI_COLS]

    run_btn = gr.Button("Evaluate Readmission Risk", variant="primary")
    out_risk = gr.Textbox(label="1. Calculated Risk Class", lines=3)
    out_metrics = gr.Textbox(value=metrics_text(THRESHOLD),
                             label="2. Held-out Test-Set Metrics at the Current Threshold",
                             lines=4, interactive=False)
    out_notice = gr.Textbox(value=DISCLAIMER, label="3. Legal Notice", interactive=False)
    outputs = [out_risk, out_metrics, out_notice]

    run_btn.click(predict, inputs=[th_slider] + ordered, outputs=outputs)
    th_slider.change(predict, inputs=[th_slider] + ordered, outputs=outputs)
    btn_fn.click(lambda t: run_preset("FN", t), inputs=[th_slider], outputs=ordered + outputs)
    btn_fp.click(lambda t: run_preset("FP", t), inputs=[th_slider], outputs=ordered + outputs)

if __name__ == "__main__":
    demo.launch()
