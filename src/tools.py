import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import re # Import re for parsing in calculation logic if moved here later

# Use absolute import (relative import also works if structure is correct)
try:
    # Import the functions needed by the tools
    from .predictive_models import predict_founder_success, calculate_market_saturation_score
    # Check if the actual model/scaler loaded successfully (optional but good)
    from .predictive_models import founder_model, founder_scaler, MODEL_LOAD_ERROR
    PREDICTIVE_MODELS_LOADED = founder_model is not None and founder_scaler is not None
    print("Predictive model functions imported successfully into tools.py.")
    if not PREDICTIVE_MODELS_LOADED:
         print(f"WARN: Founder predictive model/scaler failed to load. Tool will report errors. Detail: {MODEL_LOAD_ERROR}")
except ImportError as e:
    print(f"ERROR importing predictive_models from src: {e}")
    PREDICTIVE_MODELS_LOADED = False
    # Define dummy functions if import fails (less critical now)
    def predict_founder_success(founder_data: Dict[str, Any]) -> Optional[float]: return None
    def calculate_market_saturation_score(text: str) -> int: return 0

load_dotenv()

# --- Internal logic functions ---
# These now correctly handle None return from predict_founder_success
def _run_founder_success_tool_internal(years_exp: int, prior_exits: int, education_tier: int) -> str:
    try:
        founder_data = {
            'years_exp': int(years_exp),
            'prior_exits': int(prior_exits),
            'education_tier': int(education_tier)
        }
        probability = predict_founder_success(founder_data) # This might return None
        if probability is None:
             # Error message comes from predictive_models.py if loading failed, or from predict_founder_success if input invalid
             return f"Error: Failed to get prediction. Model/scaler may be unavailable or input invalid. Check logs."
        return f"Predicted founder success probability: {probability:.4f}"
    except ValueError as ve:
        return f"Error: Invalid input type. Ensure integers. Details: {ve}"
    except Exception as e:
        return f"An unexpected error occurred in the founder success tool internal logic: {e}"

def _run_market_saturation_tool_internal(market_analysis_text: str) -> str:
    if not isinstance(market_analysis_text, str) or len(market_analysis_text) < 10: # Reduced min length slightly
        return "Error: Insufficient or invalid text provided for market analysis (min 10 chars)."
    try:
        score = calculate_market_saturation_score(market_analysis_text)
        interpretation = "Low Saturation / Low Risk"
        if score > 65: interpretation = "High Saturation / High Risk"
        elif score > 35: interpretation = "Moderate Saturation / Medium Risk"
        # Return score AND interpretation clearly
        return f"Market Saturation Score: {score}/100 ({interpretation}). Based on keywords."
    except Exception as e:
        return f"An unexpected error occurred in the market saturation tool internal logic: {e}"

# --- Pydantic input schemas ---
class FounderInputSchema(BaseModel):
    years_exp: int = Field(description="Average years of relevant experience") # Clarified average
    prior_exits: int = Field(description="Total number of prior successful exits") # Clarified total
    education_tier: int = Field(description="Average numerical tier of education (1-3)") # Clarified average

class MarketInputSchema(BaseModel):
    market_analysis_text: str = Field(description="Text describing the market/competition")

# --- Custom tool classes ---
class FounderSuccessPredictorToolClass(BaseTool):
    name: str = "Founder Success Predictor Tool"
    description: str = (
        "Predicts the likelihood of founder success based on average experience (years), "
        "total prior successful exits, and average education tier (1-3). "
        "Input requires named arguments: years_exp (int), prior_exits (int), education_tier (int). "
        "Returns a probability score (0.0-1.0) or error message."
    )
    args_schema: Type[BaseModel] = FounderInputSchema

    def _run(self, years_exp: int, prior_exits: int, education_tier: int) -> str:
        return _run_founder_success_tool_internal(years_exp, prior_exits, education_tier)

class MarketSaturationToolClass(BaseTool):
    name: str = "Market Saturation Analysis Tool"
    description: str = (
        "Analyzes provided text describing the market/competition to calculate a saturation score (0-100) and interpretation. " # Added interpretation
        "Higher score = higher risk. Input requires named argument: 'market_analysis_text' (str)."
    )
    args_schema: Type[BaseModel] = MarketInputSchema

    def _run(self, market_analysis_text: str) -> str:
        return _run_market_saturation_tool_internal(market_analysis_text)

# --- Instantiate tools ---
# Instantiation remains the same, tool availability depends on model loading now
founder_success_tool = None
if PREDICTIVE_MODELS_LOADED: # Check if models actually loaded
    try:
        founder_success_tool = FounderSuccessPredictorToolClass()
        print("Founder Success Predictor Tool instantiated.")
    except Exception as e:
        print(f"Error instantiating FounderSuccessPredictorToolClass: {e}")
else:
     print("Founder Success Predictor Tool NOT instantiated (predictive model/scaler failed to load).")

market_saturation_tool = None
try:
    market_saturation_tool = MarketSaturationToolClass()
    print("Market Saturation Tool instantiated.")
except Exception as e:
    print(f"Error instantiating MarketSaturationToolClass: {e}")