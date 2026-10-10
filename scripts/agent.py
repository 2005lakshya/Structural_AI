import os
from dotenv import load_dotenv

load_dotenv()

import joblib
import pandas as pd

try:
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models():
    shi_model = joblib.load(os.path.join(_BASE, "models", "ml", "shi_regressor.pkl"))
    risk_model = joblib.load(os.path.join(_BASE, "models", "ml", "risk_classifier.pkl"))
    rul_model = joblib.load(os.path.join(_BASE, "models", "ml", "rul_regressor.pkl"))
    metadata = joblib.load(os.path.join(_BASE, "models", "ml", "metadata.pkl"))
    return shi_model, risk_model, rul_model, metadata


def predict_structural_metrics(inputs: dict, shi_model, risk_model, rul_model, metadata) -> dict:
    feature_cols = metadata["feature_cols"]
    risk_labels = metadata["risk_labels"]
    features = pd.DataFrame([[inputs[column] for column in feature_cols]], columns=feature_cols)

    shi = float(shi_model.predict(features)[0])
    risk_idx = int(risk_model.predict(features)[0])
    rul = float(rul_model.predict(features)[0])

    return {
        "shi": round(shi, 2),
        "risk_level": risk_labels[risk_idx],
        "rul": round(rul, 1),
        "risk_idx": risk_idx,
    }


def rule_based_recommendation(inputs: dict, predictions: dict) -> str:
    shi = predictions["shi"]
    risk = predictions["risk_level"]
    rul = predictions["rul"]
    age = inputs.get("building_age", "N/A")
    corrosion = float(inputs.get("corrosion_level", 0.0))
    crack_width = float(inputs.get("crack_width", 0.0))

    if risk == "Critical":
        severity_note = (
            "CRITICAL ALERT: The structure is in a severely deteriorated state. "
            "Immediate structural intervention is required. The building should be evaluated "
            "for potential evacuation or load restriction pending emergency repairs."
        )
    elif risk == "Moderate":
        severity_note = (
            "MODERATE CONCERN: The structure shows significant signs of deterioration. "
            "Scheduled maintenance and reinforcement is recommended within the next 6-12 months."
        )
    else:
        severity_note = (
            "SAFE: The structure is in acceptable condition. Continue routine inspection "
            "on an annual basis."
        )

    corrosion_note = ""
    if corrosion > 0.7:
        corrosion_note = "- High corrosion level detected: consider anti-corrosion treatment and rebar inspection.\n"
    elif corrosion > 0.4:
        corrosion_note = "- Moderate corrosion detected: apply protective coatings.\n"

    if crack_width > 5.0:
        crack_note = "- Wide cracks (>5mm) detected: structural crack injection and patching required.\n"
    elif crack_width > 2.0:
        crack_note = "- Medium-width cracks (2-5mm) detected: monitoring and surface sealing recommended.\n"
    else:
        crack_note = "- Minor crack widths (<2mm) detected: routine surface sealing is sufficient.\n"

    rul_note = f"- Estimated Remaining Useful Life: {rul:.1f} years.\n"
    if rul < 5:
        rul_note += "  Critical: Plan for structural replacement or major overhaul.\n"
    elif rul < 15:
        rul_note += "  Warning: Plan for major rehabilitation within 5-10 years.\n"
    else:
        rul_note += "  Acceptable service life remaining with proper maintenance.\n"

    report = f"""
==================================================
        AI STRUCTURAL HEALTH INSPECTION REPORT
==================================================

BUILDING SUMMARY
  Building Age     : {age} years
  SHI Score        : {shi:.1f} / 100
  Risk Level       : {risk}
  Est. RUL         : {rul:.1f} years

{severity_note}

DETAILED FINDINGS
{corrosion_note}{crack_note}{rul_note}
RECOMMENDED ACTIONS
  1. Conduct a detailed structural engineering survey.
  2. Review foundation integrity and rebar condition.
  3. Apply crack repair using epoxy injection where crack width > 2mm.
  4. Install structural health monitoring sensors for real-time tracking.
  5. Review load documentation and ensure compliance with updated codes.

==================================================
"""
    return report.strip()


def llm_recommendation(inputs: dict, predictions: dict, api_key: str) -> str:
    llm = ChatGoogleGenerativeAI(model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0.3, google_api_key=api_key)
    prompt = f"""You are a senior structural health monitoring engineer. Analyze the following building inspection data and provide a professional technical report with clear findings and actionable recommendations.

BUILDING DATA:
- Building Age: {inputs.get('building_age')} years
- Corrosion Level: {inputs.get('corrosion_level')} (scale 0-1)
- Crack Width: {inputs.get('crack_width')} mm
- Crack Density: {inputs.get('crack_density')} cracks/m2
- Moisture Content: {inputs.get('moisture_content')}%
- Compressive Strength: {inputs.get('compressive_strength')} MPa
- Load Stress: {inputs.get('load_stress')} MPa
- Temperature: {inputs.get('temperature')} C
- Humidity: {inputs.get('humidity')}%

ML MODEL PREDICTIONS:
- Structural Health Index (SHI): {predictions['shi']} / 100
- Risk Level: {predictions['risk_level']}
- Estimated Remaining Useful Life (RUL): {predictions['rul']} years

Write a comprehensive inspection report with sections: Executive Summary, Key Findings, Risk Assessment, and Recommended Actions."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def analyze_structure(inputs: dict) -> tuple:
    shi_model, risk_model, rul_model, metadata = load_models()
    predictions = predict_structural_metrics(inputs, shi_model, risk_model, rul_model, metadata)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and LANGCHAIN_AVAILABLE:
        print("Using LLM mode.")
        try:
            recommendation = llm_recommendation(inputs, predictions, api_key)
        except Exception as e:
            print(f"LLM failed ({e}), falling back to rule-based mode.")
            recommendation = rule_based_recommendation(inputs, predictions)
    else:
        print("Using rule-based mode.")
        recommendation = rule_based_recommendation(inputs, predictions)

    return predictions, recommendation


if __name__ == "__main__":
    sample_input = {
        "building_age": 45.0,
        "corrosion_level": 0.75,
        "crack_width": 6.2,
        "crack_density": 12.5,
        "moisture_content": 10.2,
        "compressive_strength": 24.0,
        "temperature": 32.0,
        "humidity": 65.0,
        "load_stress": 25.0,
    }
    predictions, recommendation = analyze_structure(sample_input)
    print("=== PREDICTIONS ===")
    print(predictions)
    print("\n=== RECOMMENDATION ===")
    print(recommendation)
