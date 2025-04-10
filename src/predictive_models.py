import os
import re
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

print("Loading predictive model functions...")

# Correct path calculation using __file__
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # project_root
MODEL_DIR = os.path.join(BASE_DIR, "models")
FOUNDER_MODEL_PATH = os.path.join(MODEL_DIR, "founder_success_model.keras")
FOUNDER_SCALER_PATH = os.path.join(MODEL_DIR, "founder_scaler.joblib")

print(f"DEBUG: Base directory resolved to: '{BASE_DIR}'")
print(f"DEBUG: Model directory resolved to: '{MODEL_DIR}'")
print(f"DEBUG: Attempting to use model path: '{FOUNDER_MODEL_PATH}'")
print(f"DEBUG: Attempting to use scaler path: '{FOUNDER_SCALER_PATH}'")

# --- Load Models and Scaler ---
founder_model = None
founder_scaler = None
MODEL_LOAD_ERROR = None

# Try loading scaler first
try:
    if os.path.exists(FOUNDER_SCALER_PATH):
        founder_scaler = joblib.load(FOUNDER_SCALER_PATH)
        print(f"Founder scaler loaded from {FOUNDER_SCALER_PATH}")
    else:
        MODEL_LOAD_ERROR = f"Scaler file not found at {FOUNDER_SCALER_PATH}"
        print(f"ERROR: {MODEL_LOAD_ERROR}")
except Exception as e:
    MODEL_LOAD_ERROR = f"Error loading scaler: {e}"
    print(f"ERROR: {MODEL_LOAD_ERROR}")
    founder_scaler = None

# Try loading model only if scaler loaded
if founder_scaler is not None:
    try:
        if not os.path.exists(FOUNDER_MODEL_PATH):
            error_msg = f"Model file not found at {FOUNDER_MODEL_PATH}"
            MODEL_LOAD_ERROR = f"{MODEL_LOAD_ERROR}. {error_msg}" if MODEL_LOAD_ERROR else error_msg
            print(f"ERROR: {error_msg}")
            founder_model = None
        else:
            founder_model = tf.keras.models.load_model(FOUNDER_MODEL_PATH)
            print(f"Founder success model loaded from {FOUNDER_MODEL_PATH}")
            # Clear error if only model was missing and now found
            if MODEL_LOAD_ERROR and "Model file not found" in MODEL_LOAD_ERROR and "Scaler file not found" not in MODEL_LOAD_ERROR:
                 MODEL_LOAD_ERROR = None
            elif not MODEL_LOAD_ERROR:
                 MODEL_LOAD_ERROR = None
    except Exception as e:
        error_msg = f"Error loading founder model (.keras): {e}"
        MODEL_LOAD_ERROR = f"{MODEL_LOAD_ERROR}. {error_msg}" if MODEL_LOAD_ERROR else error_msg
        print(f"ERROR: {error_msg}")
        founder_model = None
else:
    if not MODEL_LOAD_ERROR: # If scaler error wasn't set before
        MODEL_LOAD_ERROR = "Scaler failed to load, skipping model load."
    print(MODEL_LOAD_ERROR)

# --- Dummy Model Fallback Removed ---
# No dummy model creation here. Let predict_founder_success return None if loading failed.

# --- Founder Success Prediction Function ---
def predict_founder_success(founder_data: Dict[str, Any]) -> Optional[float]:
    """
    Predicts the likelihood of founder success based on input features.
    Returns None if model/scaler is unavailable or input is invalid.
    """
    global founder_model, founder_scaler, MODEL_LOAD_ERROR

    if founder_model is None or founder_scaler is None:
        print(f"Prediction failed: Model or scaler not available. Error: {MODEL_LOAD_ERROR}")
        return None # Explicitly return None

    required_keys = ['years_exp', 'prior_exits', 'education_tier']
    if not all(key in founder_data for key in required_keys):
        print(f"Prediction failed: Input data missing required keys: {required_keys}. Got: {founder_data}")
        return None

    try:
        feature_names = ['years_exp', 'prior_exits', 'education_tier']
        input_df = pd.DataFrame({
            'years_exp': [float(founder_data['years_exp'])],
            'prior_exits': [int(founder_data['prior_exits'])],
            'education_tier': [int(founder_data['education_tier'])]
        })[feature_names] # Enforce column order

        input_scaled = founder_scaler.transform(input_df)
        prediction_proba = founder_model.predict(input_scaled, verbose=0)
        success_probability = float(prediction_proba[0][0])

        # Clamp probability between 0 and 1 just in case model outputs slightly outside range
        return max(0.0, min(1.0, success_probability))

    except ValueError as ve:
        print(f"Prediction failed: Invalid data type in input. {ve}. Input: {founder_data}")
        return None
    except Exception as e:
        print(f"Error during founder success prediction: {e}. Input: {founder_data}")
        return None

# --- Market Saturation Model (Rule-Based) ---
# (No changes needed here)
SATURATION_KEYWORDS = { ... } # Keep as is
DIFFERENTIATION_KEYWORDS = { ... } # Keep as is

def calculate_market_saturation_score(text: str) -> int:
    # ... (Keep existing implementation) ...
    if not text or not isinstance(text, str) or len(text.split()) < 5: return 0
    text_lower = text.lower()
    raw_score = 0.0
    def count_occurrences(keyword, text):
        escaped_keyword = re.escape(keyword)
        pattern = r'\b' + escaped_keyword + r'\b'
        return len(re.findall(pattern, text, re.IGNORECASE))
    for keyword, weight in SATURATION_KEYWORDS.items(): raw_score += count_occurrences(keyword, text_lower) * weight
    for keyword, weight in DIFFERENTIATION_KEYWORDS.items(): raw_score += count_occurrences(keyword, text_lower) * weight
    min_expected_raw_score = -10.0; max_expected_raw_score = 30.0
    if max_expected_raw_score == min_expected_raw_score: scaled_score = 50.0
    else: scaled_score = ((raw_score - min_expected_raw_score) / (max_expected_raw_score - min_expected_raw_score)) * 100
    final_score = max(0, min(100, int(round(scaled_score))))
    return final_score

# --- Example Usage ---
if __name__ == '__main__':
    # ... (Keep existing test code) ...
    pass