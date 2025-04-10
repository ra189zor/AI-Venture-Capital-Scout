# AI Venture Capital Scout 🤖📊

An AI-powered application using CrewAI and LangChain to analyze startup pitch decks (PDFs) and provide insights for venture capital assessment.

## Overview

This tool automates the initial screening of pitch decks by:
1.  Extracting key information (TAM, Burn Rate, Founder Experience, Market Overview, Competition) from uploaded PDF documents.
2.  Utilizing AI agents (built with CrewAI) to analyze the extracted data:
    *   **Market Analyst:** Assesses market size, competition, and saturation using text analysis, web search (DuckDuckGo), and keyword-based scoring.
    *   **Team Evaluator:** Estimates founder attributes from text and uses a predictive TensorFlow model (if available) to assess team potential.
3.  Calculating overall scores for Market Potential and Founder Potential based on the agents' analysis.
4.  Presenting the findings in an interactive Streamlit dashboard.

## Technology Stack

*   **Frontend:** Streamlit
*   **AI Agent Framework:** CrewAI
*   **LLM Orchestration & Data Loading:** LangChain (`langchain`, `langchain-community`, `langchain-openai`)
*   **LLM:** OpenAI (GPT-3.5-Turbo or configurable via `.env`)
*   **Predictive Modeling:** TensorFlow, Scikit-learn, Pandas
*   **PDF Text Extraction:** `PyPDFLoader` (via `langchain-community`)
*   **Environment Variables:** `python-dotenv`

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv # Or 'env' or your preferred name
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you encounter issues with TensorFlow on certain platforms, refer to the official TensorFlow installation guide.)*

4.  **Predictive Model Files (Optional but Recommended):**
    *   This application uses a pre-trained TensorFlow model (`.keras`) and a Scikit-learn scaler (`.joblib`) to predict founder success probability.
    *   Place your trained `founder_success_model.keras` and `founder_scaler.joblib` files inside the `/models` directory in the project root.
    *   If these files are not found, the application will still run, but the Founder Success Prediction will default to N/A or a non-predictive assessment.

5.  **Configure Environment Variables:**
    *   Copy the `.env.example` file to a new file named `.env`:
        ```bash
        # Windows (Command Prompt)
        copy .env.example .env
        # Windows (PowerShell)
        Copy-Item .env.example .env
        # macOS/Linux
        cp .env.example .env
        ```
    *   Open the `.env` file and add your **OpenAI API Key**:
        ```dotenv
        OPENAI_API_KEY="sk-..." # Replace with your actual key
        OPENAI_MODEL_NAME="gpt-3.5-turbo"
        ```

## Usage

1.  **Run the Streamlit App:** Make sure your virtual environment is activated and you are in the project's root directory.
    ```bash
    streamlit run src/app.py
    ```
2.  **Upload PDF:** Use the file uploader in the web interface to select a pitch deck PDF file.
3.  **View Analysis:** The application will process the PDF, run the AI agents, calculate scores, and display the results on the dashboard.

## Limitations

*   **PDF Text Extraction:** This version uses `PyPDFLoader` for text extraction. It works well for digitally created PDFs with selectable text layers. **It cannot extract text from image-based PDFs (scans).** PDFs that are purely images will result in an extraction failure.
*   **Predictive Model:** The accuracy of the Founder Potential score depends heavily on the quality and training data of the included predictive model (`.keras` file).
*   **LLM Variability:** AI agent responses and analysis can vary slightly between runs due to the nature of Large Language Models.
*   **Context Window:** Very long documents might still exceed the LLM's context window during the entity extraction phase, even with truncation enabled, potentially leading to errors or incomplete extraction.

## Deployment to Streamlit Cloud

1.  **Push to GitHub:** Ensure your `.gitignore` file includes `.env` and commit your code (including `.env.example`) to a GitHub repository.
2.  **Connect Streamlit Cloud:** Link your Streamlit Cloud account to your GitHub repository.
3.  **Deploy:** Choose the repository and branch to deploy.
4.  **Add Secrets:** In your app's settings on Streamlit Cloud (**Settings -> Secrets**), add your `OPENAI_API_KEY`:
    ```toml
    OPENAI_API_KEY="sk-..."
    ```
    Streamlit Cloud will securely inject this as an environment variable.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

                                                                               Made By AB

