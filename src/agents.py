import os
import json
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel, Field

# --- Imports and Tool Setup ---
# Use absolute import for tools
try:
    from src.tools import founder_success_tool, market_saturation_tool
    # Import the prediction function directly for use in run_crew_analysis
    from src.predictive_models import predict_founder_success
    print("Custom tools and prediction function imported successfully.")
    FOUNDER_TOOL_AVAILABLE = founder_success_tool is not None
    MARKET_TOOL_AVAILABLE = market_saturation_tool is not None
    # Check if the actual prediction function is usable (models loaded)
    from src.predictive_models import founder_model, founder_scaler
    PREDICTION_POSSIBLE = founder_model is not None and founder_scaler is not None
    print(f" Founder Tool Available: {FOUNDER_TOOL_AVAILABLE}")
    print(f" Market Tool Available: {MARKET_TOOL_AVAILABLE}")
    print(f" Prediction Possible (Models Loaded): {PREDICTION_POSSIBLE}")
except ImportError as e:
    print(f"FATAL ERROR importing tools/models: {e}")
    FOUNDER_TOOL_AVAILABLE = False
    MARKET_TOOL_AVAILABLE = False
    PREDICTION_POSSIBLE = False
    founder_success_tool = None
    market_saturation_tool = None
    def predict_founder_success(*args, **kwargs): return None # Dummy if import fails

# DuckDuckGo Tool Wrapper (keep as is)
# ... SearchInput, DuckDuckGoSearchTool class ...
search_tool = None
SEARCH_TOOL_AVAILABLE = False
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    duckduckgo_search_run = DuckDuckGoSearchRun()
    class SearchInput(BaseModel): query: str = Field(description="The search query string")
    class DuckDuckGoSearchTool(BaseTool):
        name: str = "DuckDuckGo Search"; description: str = "..."; args_schema: Type[BaseModel] = SearchInput
        def _run(self, query: str) -> str: return duckduckgo_search_run.run(query)
    search_tool = DuckDuckGoSearchTool()
    SEARCH_TOOL_AVAILABLE = True
    print("Search tool loaded and wrapped.")
except ImportError:
    print("Warning: DuckDuckGoSearchRun import failed."); search_tool = None; SEARCH_TOOL_AVAILABLE = False

# --- LLM Config ---
# ... (keep as is) ...
llm_instance = None
# ... (LLM initialization logic) ...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
if OPENAI_API_KEY:
    try:
        llm_instance = ChatOpenAI(model=OPENAI_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.3, max_tokens=1000)
        print(f"ChatOpenAI LLM instance created with model: {OPENAI_MODEL_NAME}")
    except Exception as e: print(f"FATAL ERROR creating ChatOpenAI: {e}"); llm_instance = None
else: print("FATAL ERROR: OPENAI_API_KEY not found.")


# --- Agent Definitions ---
# ... (Market Analyst definition remains the same) ...
market_analyst = None
team_evaluator = None

if llm_instance:
    market_analyst_tools = []
    if SEARCH_TOOL_AVAILABLE: market_analyst_tools.append(search_tool)
    if MARKET_TOOL_AVAILABLE: market_analyst_tools.append(market_saturation_tool)
    # Team evaluator doesn't need the tool directly anymore
    team_evaluator_tools = []

    try:
        market_analyst = Agent(role="Startup Market Analyst", goal="...", backstory="...", tools=market_analyst_tools, llm=llm_instance, verbose=True, allow_delegation=False)
        print("DEBUG: Market Analyst agent created.")

        # Team Evaluator - No tools needed directly
        print(f"\nDEBUG: Tools for Team Evaluator: {[t.name for t in team_evaluator_tools]}") # Should be empty
        team_evaluator = Agent(
            role="Founder Team Information Extractor", # Changed role slightly
            goal="Read the provided 'Founder Experience Summary' text and extract integer estimates for average experience years, total prior exits, and average education tier.", # Simplified goal
            backstory="You are an assistant focused on extracting specific numerical data points from text about founding teams. You provide only the requested estimates.",
            tools=team_evaluator_tools, # Empty list
            llm=llm_instance,
            verbose=True,
            allow_delegation=False
        )
        print("DEBUG: Team Evaluator agent created.")

    except Exception as agent_creation_error:
        print(f"\nFATAL ERROR during Agent creation: {agent_creation_error}")
        market_analyst = None; team_evaluator = None
else:
    print("ERROR: LLM Instance not created. Cannot define Agents.")


# --- Task Definitions (Team Task Simplified) ---
market_analysis_task = None
team_evaluation_task = None

if market_analyst and team_evaluator:
    print("\nDEBUG: All required agents seem created. Defining tasks...")
    try:
        # Market Analysis Task (keep as is from previous version)
        market_analysis_task = Task(
            description=(
                "**Objective:** Provide a VC-ready analysis...\n"
                # ... (rest of description using placeholders) ...
            ),
            expected_output=(
                "**Market Analysis Report:**\n\n"
                # ... (rest of expected output) ...
            ),
            agent=market_analyst
        )
        print("DEBUG: Market analysis task defined.")

        # Team Evaluation Task - SIMPLIFIED - Agent only extracts estimates
        team_evaluation_task = Task(
            description=(
                "**Objective:** Extract founder attribute estimates from the text.\n"
                "**Steps:**\n"
                "1. Carefully read the 'Founder Experience Summary' text provided.\n"
                "2. Based *only* on reading the text, provide your best *integer* estimates for:\n"
                "    a. Average years of relevant experience.\n"
                "    b. Total number of prior *successful* exits mentioned.\n"
                "    c. An overall average education tier (1=Basic/NA, 2=Standard, 3=Advanced/TopTier).\n"
                "3. Output *only* these three estimates in a structured format. If you cannot estimate a value from the text, use the word 'Unknown'.\n"
                "\n**--- CONTEXT PROVIDED ---**\n"
                "Founder Experience Summary Text: {founder_summary}"
            ),
            expected_output=( # Agent just outputs the numbers or "Unknown"
                "**Founder Attribute Estimates:**\n"
                "Average Experience Years: [Integer or Unknown]\n"
                "Total Prior Exits: [Integer or Unknown]\n"
                "Average Education Tier: [1, 2, 3 or Unknown]"
            ),
            agent=team_evaluator # Agent assigned to extract
        )
        print("DEBUG: Team evaluation task defined.")

    except Exception as e:
        print(f"Error defining tasks: {e}")
        market_analysis_task = None; team_evaluation_task = None
else:
     print("\nERROR: One or more required Agents failed. Cannot define Tasks.")

# --- Crew Definition ---
# ... (keep as is) ...
pitch_deck_analysis_crew = None
if market_analyst and team_evaluator and market_analysis_task and team_evaluation_task:
    print("\nDEBUG: Defining Crew...")
    try:
        pitch_deck_analysis_crew = Crew(agents=[market_analyst, team_evaluator], tasks=[market_analysis_task, team_evaluation_task], process=Process.sequential, verbose=True)
        print("DEBUG: Crew defined successfully.")
    except Exception as e: print(f"Error defining crew: {e}"); pitch_deck_analysis_crew = None
else: print("\nERROR: Missing agents/tasks. Cannot define Crew.")


# --- Function to Run the Crew (Handles Tool Call & Report Generation) ---
def run_crew_analysis(extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Runs the crew, attempts prediction based on agent output, and formats results."""
    if not pitch_deck_analysis_crew:
        print("ERROR: Crew not defined.")
        return {"error": "Crew not initialized."}
    if not extracted_data:
         print("ERROR: No extracted data provided.")
         return {"error": "No extracted data provided."}

    try:
        inputs = { # Prepare inputs for kickoff
            'market_overview_text': extracted_data.get('Market Overview', 'Not Provided'),
            'competition_text': extracted_data.get('Competition', 'Not Provided'),
            'tam_extracted': extracted_data.get('TAM', 'Not Provided'),
            'competitors_extracted': ", ".join(extracted_data.get('Key Competitors', [])) if extracted_data.get('Key Competitors') else 'Not Provided',
            'founder_summary': extracted_data.get('Founder Experience', 'Not Provided'),
        }
        print("\nDEBUG: Inputs for crew.kickoff():", json.dumps(inputs, indent=2))

        result = pitch_deck_analysis_crew.kickoff(inputs=inputs)

        # --- Post-Analysis Processing ---
        market_report = ""
        team_estimates_report = "" # Agent now outputs estimates
        final_team_assessment = "Assessment not generated." # Default
        calculated_market_score = None
        calculated_founder_score_prob = None # Probability 0-1

        # Safely access task outputs
        if result and result.tasks_output:
             if len(result.tasks_output) > 0 and hasattr(result.tasks_output[0], 'raw'):
                 market_report = result.tasks_output[0].raw
             if len(result.tasks_output) > 1 and hasattr(result.tasks_output[1], 'raw'):
                 team_estimates_report = result.tasks_output[1].raw # Get the estimates string
        else:
             print("Warning: Crew result or tasks_output is missing or empty.")

        print("\nDEBUG: Raw Market Report:\n", market_report)
        print("\nDEBUG: Raw Team Estimates Report:\n", team_estimates_report)

        # Calculate Market Score (keep previous logic)
        try:
            # ... (keep market score calculation logic based on market_report and extracted_data) ...
            saturation_score_match = re.search(r"Market Saturation Score:\s*(\d+)/100", market_report, re.IGNORECASE)
            saturation_score_num = int(saturation_score_match.group(1)) if saturation_score_match else None
            tam_text = extracted_data.get('TAM', '').lower()
            is_large_tam = 'billion' in tam_text or ('million' in tam_text and int(re.sub(r'[^\d.]', '', tam_text).split('.')[0]) > 500)
            base_score = 50
            if is_large_tam: base_score += 15
            if saturation_score_num is not None:
                if saturation_score_num < 35: base_score += 15
                elif saturation_score_num <= 65: base_score += 5
                else: base_score -= 15
            calculated_market_score = max(0, min(100, base_score))
            print(f"DEBUG: Calculated Market Score: {calculated_market_score}")
        except Exception as e:
            print(f"Error calculating market score: {e}"); calculated_market_score = None

        # --- Process Team Estimates and Attempt Prediction ---
        exp_years = None
        prior_exits = None
        edu_tier = None
        prediction_result_text = "Prediction N/A (Could not extract estimates)." # Default

        # Try parsing estimates from the agent's output
        try:
            exp_match = re.search(r"Average Experience Years:\s*(\d+)", team_estimates_report, re.IGNORECASE)
            exits_match = re.search(r"Total Prior Exits:\s*(\d+)", team_estimates_report, re.IGNORECASE)
            edu_match = re.search(r"Average Education Tier:\s*([1-3])", team_estimates_report, re.IGNORECASE) # Only 1, 2, or 3

            if exp_match: exp_years = int(exp_match.group(1))
            if exits_match: prior_exits = int(exits_match.group(1))
            if edu_match: edu_tier = int(edu_match.group(1))

            print(f"DEBUG: Parsed Estimates - Exp: {exp_years}, Exits: {prior_exits}, Edu: {edu_tier}")

            # If all estimates parsed AND prediction is possible (models loaded)
            if exp_years is not None and prior_exits is not None and edu_tier is not None and PREDICTION_POSSIBLE:
                print("DEBUG: Attempting prediction with parsed estimates...")
                founder_data_input = {
                    'years_exp': exp_years,
                    'prior_exits': prior_exits,
                    'education_tier': edu_tier
                }
                # Call the prediction function directly
                probability = predict_founder_success(founder_data_input)

                if probability is not None:
                    calculated_founder_score_prob = probability # Store the probability
                    prediction_result_text = f"Predicted founder success probability: {probability:.4f}"
                    print(f"DEBUG: Prediction successful: {calculated_founder_score_prob}")
                else:
                    prediction_result_text = "Prediction N/A (Model execution failed)."
                    print("DEBUG: Prediction function returned None.")
            elif not PREDICTION_POSSIBLE:
                 prediction_result_text = "Prediction N/A (Predictive model not loaded)."
            else:
                 # Estimates weren't all parsed
                 prediction_result_text = "Prediction N/A (Could not estimate all inputs from text)."

        except Exception as e:
            print(f"Error parsing team estimates or running prediction: {e}")
            prediction_result_text = "Prediction N/A (Error during processing)."

        # --- Generate Final Team Assessment Text ---
        # Combine the original summary, estimates, and prediction result
        founder_summary_text = extracted_data.get('Founder Experience', 'Not Provided')
        qualitative_summary = "Team assessment based on available data." # Simple default
        # Could potentially ask another LLM call here for a better summary, but keeping it simple:
        if calculated_founder_score_prob is not None:
             if calculated_founder_score_prob > 0.65: qualitative_summary = "Analysis suggests a strong founding team based on experience profile and prediction."
             elif calculated_founder_score_prob > 0.4: qualitative_summary = "Analysis suggests a moderately capable founding team."
             else: qualitative_summary = "Analysis indicates potential challenges or lack of demonstrated experience for the founding team."
        elif "Unknown" in team_estimates_report or "Could not estimate" in prediction_result_text:
             qualitative_summary = "Insufficient information in the provided text to fully assess the team or run prediction."
        else: # Models not loaded or other error
             qualitative_summary = "Team assessment relies on text only; predictive model unavailable or failed."


        final_team_assessment = (
            f"**Founder Experience Summary:**\n{founder_summary_text}\n\n"
            f"**Extracted Estimates:**\n{team_estimates_report}\n\n" # Show what the agent extracted
            f"**Prediction Result:** {prediction_result_text}\n\n"
            f"**Qualitative Summary:** {qualitative_summary}"
        )

        # Structure the final output dictionary
        output = {
            "market_analysis": market_report,
            "team_assessment": final_team_assessment, # Use the generated report
            "market_score": calculated_market_score,
            "founder_score": calculated_founder_score_prob # Probability 0-1
        }
        print("\nDEBUG: Final Crew Analysis Output Dict:", json.dumps(output, indent=2))
        return output

    except Exception as e:
        print(f"Error running crew analysis: {e}")
        return {"error": f"Crew execution failed: {e}"}