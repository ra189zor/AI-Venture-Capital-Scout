# src/app.py

import streamlit as st
import os
import time
import json
import re
from typing import Optional

# --- Load Environment Variables ---
# This needs to happen early, before checking the password
from dotenv import load_dotenv
# Correct path calculation using __file__ relative to this script (in src/)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Attempted loading .env from: {dotenv_path}")
# --- End Load ---

# Set page config first
st.set_page_config(page_title="AI VC Scout", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (Keep your existing CSS)
st.markdown("""
<style>
    /* Your existing CSS styles */
    /* Dark theme background for the entire app */
    .stApp { background-color: #1E1E1E; color: #E0E0E0; }
    .main-title { font-size: 42px; font-weight: 700; color: #5D8BF4; text-align: center; margin-bottom: 0px; }
    .subtitle { font-size: 20px; color: #BBBBBB; text-align: center; margin-bottom: 20px; }
    .metric-card { background-color: #2D2D2D; color: #E0E0E0; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); margin-bottom: 15px; }
    .stExpander { border-radius: 8px; border: 1px solid #444; background-color: #2D2D2D; margin-top: 15px;}
    .stExpander > details > summary {font-weight: bold;}
    .stProgress > div > div { border-radius: 10px; background-color: #5D8BF4; }
    .stProgress > div > div > div { background-color: #5D8BF4; }
    .success-message { background-color: #143601; color: #9AE66E; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0; }
    .info-message { background-color: #01303f; color: #7CD5F9; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0; }
    .error-message { background-color: #3F0101; color: #F97C7C; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0; }
    /* Widget styling */
    .stButton>button { background-color: #5D8BF4; color: #FFFFFF; border: none; border-radius: 5px; padding: 8px 16px;}
    .stButton>button:hover { background-color: #4a7bdc; }
    .stTextInput>div>div>input, .stSelectbox>div>div>select { background-color: #3D3D3D; color: #E0E0E0; border: 1px solid #444; }
    /* File uploader */
    .stFileUploader>label>div>button { background-color: #5D8BF4; color: #FFFFFF; border: none; border-radius: 5px; }
    .stFileUploader>label { color: #E0E0E0 !important; }
    .stFileUploader>div>div { color: #BBBBBB; }
    /* Divider */
    hr { border-color: #444; }
    /* Sidebar */
    .css-1d391kg { background-color: #1E1E1E; }
    .sidebar-content { background-color: #2D2D2D; padding: 20px; border-radius: 10px; margin-top: 20px; color: #E0E0E0; }
    .sidebar-content h3 { color: #5D8BF4; text-align: center; }
    .sidebar-content p { text-align: center; font-size: 0.9em; }
    .sidebar-content ol { text-align: left; margin-left: 20px; font-size: 0.85em; }
    .sidebar-content ul { list-style-type: none; padding: 0; text-align: center; }
    .sidebar-content li { margin-bottom: 8px; font-size: 0.85em;}
    .sidebar-content hr { border-color: #444; }
    .sidebar-content a { color: #7CD5F9; }
</style>
""", unsafe_allow_html=True)

# --- Import Backend Functions ---
# Defer imports until after password check if they are slow or rely on secrets implicitly,
# but usually fine here if they only define functions.
try:
    from data_pipeline import run_pipeline as run_data_pipeline
    from agents import run_crew_analysis
    BACKEND_LOADED = True
    print("Backend functions (data_pipeline, agents) loaded successfully.")
except ImportError as e:
    # Show error even before password check if backend fails to load
    st.error(f"Fatal Error: Failed to import backend modules. Check structure and errors in console. Details: {e}")
    # import traceback
    # st.error(traceback.format_exc()) # Show full traceback in UI for debug
    BACKEND_LOADED = False
    st.stop() # Stop execution if backend fails

# --- Define Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "temp_uploads")
ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "analysis_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# --- Parsing Helper Functions ---
def parse_founder_score(crew_output: Optional[dict]) -> Optional[float]:
    """Gets the Founder Score (probability 0-1) from the crew output dictionary."""
    if not crew_output or not isinstance(crew_output, dict):
        return None
    score = crew_output.get("founder_score")
    if score is not None:
        try:
            return max(0.0, min(1.0, float(score)))
        except (ValueError, TypeError):
            print(f"Warning: Could not parse founder_score '{score}' as float.")
            return None
    return None

def parse_market_score(crew_output: Optional[dict]) -> Optional[int]:
    """Gets the Market Score (0-100) from the crew output dictionary."""
    if not crew_output or not isinstance(crew_output, dict):
        return None
    score = crew_output.get("market_score")
    if score is not None:
        try:
            return max(0, min(100, int(score)))
        except (ValueError, TypeError):
            print(f"Warning: Could not parse market_score '{score}' as int.")
            return None
    return None

# --- Password Check Function ---
def check_password():
    """Returns True if the user had entered the correct password."""

    # Get the correct password from environment variables/secrets
    correct_password = os.getenv("APP_PASSWORD")
    if not correct_password:
        # If no password is set in secrets/env, allow access (or handle as error)
        st.warning("Warning: APP_PASSWORD environment variable not set. Access granted without password.")
        return True # Or False if you want to force password setup

    # Use session state to store password check status
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    # If already authenticated, return True
    if st.session_state.password_correct:
        return True

    # Show password input form using columns for better layout
    st.markdown('<h1 class="main-title">🔑 Enter Password</h1>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) # Add some space

    col1, col2, col3 = st.columns([1, 2, 1]) # Center the input area
    with col2:
        with st.form("password_form"):
            password_input = st.text_input("Password", type="password", key="password_input_field")
            submitted = st.form_submit_button("Login")

            if submitted:
                if password_input == correct_password:
                    st.session_state.password_correct = True
                    st.rerun() # Rerun to clear form and show main app
                else:
                    st.error("😕 Incorrect password")

    return False # Password not yet entered correctly

# --- Main Application Function ---
def run_main_app():
    """Contains the main UI and logic after password check."""

    st.markdown('<h1 class="main-title">🤖 AI Venture Capital Scout</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Automated Pitch Deck Analysis</p>', unsafe_allow_html=True)
    st.markdown("Upload a pitch deck (PDF) to initiate the automated analysis workflow.")

    uploaded_file = st.file_uploader(
        "Choose a pitch deck file:",
        type=["pdf"],
        accept_multiple_files=False,
        key="file_uploader"
    )

    # Initialize session state keys (safe to do here)
    default_state = {
        "analysis_complete": False, "extracted_data": None, "crew_report_dict": None,
        "error_message": None, "info_message": None, "processing": False,
        "last_processed_filename": None, "founder_score_prob": None, "market_score_100": None
    }
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Analysis Workflow Trigger ---
    initiate_analysis = False
    if uploaded_file is not None and not st.session_state.processing:
        if st.session_state.last_processed_filename != uploaded_file.name:
            initiate_analysis = True
            print(f"New file detected: {uploaded_file.name}. Resetting state.")
            # Reset only analysis-specific state, keep password state
            for key, default_value in default_state.items():
                 if key != "password_correct": # Don't reset password status
                     st.session_state[key] = default_value
            st.session_state.last_processed_filename = uploaded_file.name
            st.session_state.info_message = f"Preparing to analyze '{uploaded_file.name}'..."

    if initiate_analysis:
        st.session_state.processing = True
        st.session_state.error_message = None
        temp_file_path = None

        st.markdown(f'<div class="info-message">{st.session_state.info_message}</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 1. Save Uploaded File
            status_text.info("💾 Saving uploaded file...")
            file_extension = os.path.splitext(uploaded_file.name)[1]
            temp_filename = f"upload_{uploaded_file.name.split('.')[0]}_{int(time.time())}{file_extension}"
            temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)
            with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            print(f"File saved temporarily to: {temp_file_path}")
            progress_bar.progress(10)

            # 2. Run Data Pipeline
            status_text.info("📄 Phase 1: Extracting key data points...")
            extracted_data = run_data_pipeline(temp_file_path, output_directory=ANALYSIS_OUTPUT_DIR)
            progress_bar.progress(50)

            if extracted_data is None:
                st.session_state.error_message = "Failed during data extraction (Phase 1). Check console logs."
                raise ValueError(st.session_state.error_message)
            elif not extracted_data:
                 st.session_state.info_message = "Data extraction finished, but no specific entities found. Analysis may be limited."
                 st.session_state.extracted_data = {}
            else:
                 st.session_state.extracted_data = extracted_data
                 st.markdown('<div class="success-message">✅ Data extraction complete.</div>', unsafe_allow_html=True)

            # 3. Run Agent Crew Analysis
            status_text.info("🤖 Phase 2: AI agents analyzing data...")
            crew_output_dict = run_crew_analysis(st.session_state.extracted_data)
            progress_bar.progress(90)

            if isinstance(crew_output_dict, dict) and "error" in crew_output_dict:
                st.session_state.error_message = f"AI crew analysis failed (Phase 2): {crew_output_dict['error']}"
                st.session_state.analysis_complete = False
                st.warning(st.session_state.error_message)
            elif crew_output_dict is None:
                 st.session_state.error_message = "AI crew analysis failed unexpectedly (Phase 2)."
                 st.session_state.analysis_complete = False
                 st.error(st.session_state.error_message)
            else:
                st.session_state.crew_report_dict = crew_output_dict
                st.markdown('<div class="success-message">✅ AI analysis complete!</div>', unsafe_allow_html=True)
                st.session_state.analysis_complete = True
                st.session_state.founder_score_prob = parse_founder_score(crew_output_dict)
                st.session_state.market_score_100 = parse_market_score(crew_output_dict)
                print(f"Parsed Scores - Founder Probability: {st.session_state.founder_score_prob}, Market Score: {st.session_state.market_score_100}")

            progress_bar.progress(100)
            status_text.success("🏁 Analysis workflow finished.")
            time.sleep(2)

        except Exception as e:
            st.session_state.error_message = f"An unexpected error occurred during processing: {e}"
            st.error(st.session_state.error_message)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as e: st.warning(f"Could not remove temp file: {e}")
            st.session_state.processing = False
            progress_bar.empty(); status_text.empty()
            st.rerun()

    # --- Display Results Area ---
    if st.session_state.last_processed_filename:
        st.divider()
        st.markdown(f'<h2 style="text-align: center; color: #5D8BF4;">📊 Analysis Dashboard: {st.session_state.last_processed_filename}</h2>', unsafe_allow_html=True)

        if st.session_state.error_message:
            st.markdown(f'<div class="error-message">{st.session_state.error_message}</div>', unsafe_allow_html=True)

        if st.session_state.analysis_complete or st.session_state.extracted_data:
            col1, col2 = st.columns(2)
            with col1:
                with st.container():
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("#### 👥 Founder & Team Potential")
                    if st.session_state.founder_score_prob is not None:
                        st.metric(label="Predicted Success Likelihood", value=f"{st.session_state.founder_score_prob:.1%}")
                        st.caption("Calculated based on AI report & model")
                    else:
                        st.warning("Founder Score: N/A", icon="⚠️")
                        st.caption("Score could not be calculated.")
                    st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                with st.container():
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("#### 📈 Market Analysis Score")
                    if st.session_state.market_score_100 is not None:
                        st.metric(label="Market Score", value=f"{st.session_state.market_score_100} / 100")
                        st.progress(st.session_state.market_score_100 / 100)
                        risk_level = "High Potential" if st.session_state.market_score_100 > 65 else ("Medium Potential" if st.session_state.market_score_100 > 35 else "Lower Potential")
                        color = "green" if risk_level == "High Potential" else ("orange" if risk_level == "Medium Potential" else "red")
                        st.markdown(f"<small style='color:{color};'>Interpretation: {risk_level}</small>", unsafe_allow_html=True)
                        st.caption("Calculated based on AI report & data")
                    else:
                        st.warning("Market Score: N/A", icon="⚠️")
                        st.caption("Score could not be calculated.")
                    st.markdown('</div>', unsafe_allow_html=True)

        # Display expanders for details
        if isinstance(st.session_state.crew_report_dict, dict) and \
           (st.session_state.crew_report_dict.get("market_analysis") or st.session_state.crew_report_dict.get("team_assessment")):
            with st.expander("🔍 View Full AI Agent Analysis Report"):
                market_analysis_report = st.session_state.crew_report_dict.get("market_analysis", "")
                if market_analysis_report:
                    st.markdown("### Market Analysis Report")
                    st.markdown(f"```markdown\n{market_analysis_report}\n```")
                team_assessment_report = st.session_state.crew_report_dict.get("team_assessment", "")
                if team_assessment_report:
                    st.markdown("### Founder Team Assessment")
                    st.text_area("Team Assessment Details", value=team_assessment_report, height=350, disabled=True, key="team_assessment_area")
        elif st.session_state.analysis_complete:
             st.warning("AI Crew analysis complete, but reports are missing or empty.")

        if st.session_state.extracted_data:
            with st.expander("📄 View Raw Extracted Data (JSON)"): st.json(st.session_state.extracted_data)

        # --- Clear Button ---
        st.divider()
        if st.button("Clear Results & Analyze New File"):
            print("Clear button clicked. Resetting state.")
            # Keep password_correct, clear others
            keys_to_clear = list(default_state.keys()) + ["password_input_field", "file_uploader"]
            for key in keys_to_clear:
                 if key != "password_correct" and key in st.session_state:
                     del st.session_state[key]
            st.rerun()

    # --- Sidebar (Should only show if authenticated) ---
    st.sidebar.markdown("""
    <div class="sidebar-content">
        <h3>AI VC Scout</h3>
        <p>Automated pitch deck analysis.</p>
        <hr>
        <h4>Workflow:</h4>
        <ol>
            <li>Upload PDF</li>
            <li>Extract Data</li>
            <li>Analyze (AI Agents)</li>
            <li>Calculate Scores</li>
            <li>View Report</li>
        </ol>
        <hr>
        <h4>Powered By:</h4>
        <ul>
            <li>Streamlit</li>
            <li>LangChain</li>
            <li>CrewAI</li>
            <li>OpenAI</li>
            <li>TensorFlow</li>
        </ul>
        <hr>
        <p style="font-size: 0.8em;">© 2025 AB</p>
    </div>
    """, unsafe_allow_html=True)


# --- Main Execution Logic ---
if check_password():
    # If password check passes, run the main application
    run_main_app()
# If check_password() returns False, it handles displaying the password form itself.