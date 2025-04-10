# src/data_pipeline.py (Reverted to PyPDFLoader)

import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Use standard LangChain PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# LangChain & OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain

# --- Configuration ---
load_dotenv()

# --- Constants ---
# Text limit safeguard for GPT-3.5 Turbo (16k context)
MAX_EXTRACTION_CHARS = 45000
EXTRACTION_MAX_OUTPUT_TOKENS = 512

# Schema (Simplified Founder Experience)
SCHEMA = {
    "properties": {
        "TAM": {"type": "string", "description": "Total Addressable Market value/description."},
        "Burn Rate": {"type": "string", "description": "Reported cash burn rate."},
        "Founder Experience": {"type": "string", "description": "Single text block summarizing founders' backgrounds."},
        "Market Overview": {"type": "string", "description": "General description of the market."},
        "Competition": {"type": "string", "description": "Description of the competitive landscape."},
        "Key Competitors": {"type": "array", "items": {"type": "string"}, "description": "List of specific competitor names."},
    },
    "required": []
}

# --- Initialize OpenAI LLM ---
def initialize_llm() -> Optional[ChatOpenAI]:
    """Initializes the ChatOpenAI LLM."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not found.")
        return None
    try:
        llm = ChatOpenAI(
            model=model_name, openai_api_key=openai_api_key,
            temperature=0.1, max_tokens=EXTRACTION_MAX_OUTPUT_TOKENS
        )
        print(f"Initialized ChatOpenAI with model: {model_name}")
        return llm
    except Exception as e:
        print(f"ERROR: Failed to initialize ChatOpenAI: {e}")
        return None

# --- Extract Text from PDF using PyPDFLoader ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text using PyPDFLoader."""
    try:
        print(f"Processing PDF with PyPDFLoader: {pdf_path}")
        loader = PyPDFLoader(pdf_path, extract_images=False) # Standard loader
        documents = loader.load()
        if not documents:
             print("Warning: PyPDFLoader returned no documents.")
             return ""
        # Join page content, ensuring None pages are handled
        full_text = "\n\n--- PAGE BREAK ---\n\n".join([doc.page_content for doc in documents if doc.page_content])
        print(f"Successfully extracted text using PyPDFLoader. Total chars: {len(full_text)}")
        if not full_text.strip():
             print("Warning: Extracted text is empty or only whitespace.")
        return full_text
    except Exception as e:
        print(f"Error extracting text from PDF using PyPDFLoader: {e}")
        # Optionally add import traceback; traceback.print_exc() here for more detail
        return "" # Return empty string on failure

# --- Run Data Pipeline ---
def run_pipeline(pdf_path: str, output_directory: str) -> Optional[Dict[str, Any]]:
    """Orchestrates PDF text extraction (PyPDFLoader) and LLM entity extraction."""
    print(f"\n--- Starting Data Pipeline for: {pdf_path} ---")

    # 1. Extract Text using PyPDFLoader
    text = extract_text_from_pdf(pdf_path)
    if not text: # Check if text extraction yielded anything
        print("Pipeline halted: Text extraction failed or produced no text.")
        return None # Return None if extraction fails

    # 2. Apply Text Limit (Truncation)
    original_length = len(text)
    if original_length > MAX_EXTRACTION_CHARS:
        print(f"Warning: Input text length ({original_length} chars) exceeds limit ({MAX_EXTRACTION_CHARS}). Truncating.")
        text = text[:MAX_EXTRACTION_CHARS]
        print(f"Truncated text length: {len(text)} chars.")
    else:
        print(f"Text extraction successful. Length: {original_length} chars (within limit).")

    # 3. Initialize LLM
    llm = initialize_llm()
    if not llm:
        print("Pipeline halted: LLM initialization failed.")
        return None

    # 4. Extract Entities
    try:
        print(f"Running extraction using model: {llm.model_name}")
        chain = create_extraction_chain(schema=SCHEMA, llm=llm)
        result = chain.invoke({"input": text}) # Use extracted text

        # Parse result (same logic as before)
        extracted_data = {}
        if isinstance(result, dict) and 'text' in result:
             data_content = result['text']
             if isinstance(data_content, list) and data_content and isinstance(data_content[0], dict): extracted_data = data_content[0]
             elif isinstance(data_content, dict): extracted_data = data_content
        elif isinstance(result, dict) and any(key in result for key in SCHEMA['properties']): extracted_data = {k: result.get(k) for k in SCHEMA['properties'] if k in result}

        if not extracted_data:
             print("Warning: Extraction chain returned empty or unparsed result.")
             print(f" Raw result: {result}")
             extracted_data = {} # Return empty dict, not None
        else:
             print(f"Successfully extracted entities: {list(extracted_data.keys())}")

        # 5. Optional: Save results (code removed, but output_directory is still passed)
        # ...

        print(f"--- Data Pipeline finished for: {pdf_path} ---")
        return extracted_data # Return the dictionary (even if empty)

    except Exception as e:
        print(f"Error during entity extraction: {e}")
        if "maximum context length" in str(e): print(f"ERROR: OpenAI context length error. Input length (chars): {len(text)}")
        return None # Return None on critical extraction error

# --- Example Usage ---
if __name__ == "__main__":
    # load_dotenv() # Ensure .env loaded if running directly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sample_pdf_path = os.path.join(project_root, "data", "pitch_decks", "sample.pdf") # Adjust filename
    sample_output_dir = os.path.join(project_root, "data", "analysis_outputs")

    if os.path.exists(sample_pdf_path):
        result = run_pipeline(sample_pdf_path, sample_output_dir)
        if result is not None: # Check if pipeline returned data (even empty dict)
            print("\nExtracted Data:", json.dumps(result, indent=2))
        else:
            print("\nPipeline execution failed critically (returned None).")
    else:
        print(f"Sample PDF not found at {sample_pdf_path}")