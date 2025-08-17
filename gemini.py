import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    raise ValueError("GENAI_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.5-pro"

# Give response in JSON format
generation_config = genai.types.GenerationConfig(
    response_mime_type="application/json"
)


# Store chat sessions for both parsing and answering
parse_chat_sessions = {}

# Get or create a persistent chat session for a given session_id.
async def get_chat_session(sessions_dict, session_id, system_prompt, model_name=MODEL_NAME):
    if session_id not in sessions_dict:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,   # defaults for the whole chat
            system_instruction=system_prompt       # put your system prompt here
        )        
        chat = model.start_chat(history=[])
        sessions_dict[session_id] = chat    
    return sessions_dict[session_id]

# ------------------------
# PARSE QUESTION FUNCTION
# ------------------------
async def parse_question_with_llm(question_text=None, uploaded_files=None, session_id="default_parse", retry_message=None, folder="uploads"):
    """
    Parse question with persistent chat session.
    - If retry_message is provided, sends only that to continue conversation.
    """

    SYSTEM_PROMPT = f"""
You are an AI Python code generator for multi-step data analysis and processing.

## Core Behavior
1. Break every problem into sequential steps.
2. After each step:
   - Save all intermediate findings, extracted data, or context to {folder}/metadata.txt (append mode).
3. Save only the final verified answer in {folder}/result.txt (or {folder}/result.json if a structured format is requested).

## Resources
- Primary LLM: Google Gemini
- API Key: read from environment variable (never include or print credentials)
- Working Folder: {folder}

## Capabilities
- Fetch & parse data from: URLs (HTML, JSON, CSV, APIs), databases, and local files (CSV, Excel, PDF, images, text, JSON).
- Scrape websites and summarize findings.
- Execute Python code for data processing, visualization, or querying.
- Programmatically call the Gemini API (with provided key).
- Save all intermediate and final outputs in the working folder.

## Execution Rules
1. Always return **valid JSON** in this format:
   {{
       "code": "<python_code_that_can_run_in_python_REPL>",
       "libraries": ["list", "of", "external_libraries"],
       "run_this": 1 or 0
   }}
   - `run_this=1` → I should execute this code immediately and return you the output.  
   - `run_this=0` → No execution needed (final verified code or step complete).  
2. Do **not** return explanations — JSON only.  
3. If an error occurs and I provide you the error message, return **corrected code** only. If repeated errors occur, generate fresh new code.  
4. The **final step** must always save the definitive answer in {folder}/result.txt (or {folder}/result.json if applicable).  

## Notes
- Always prefer incremental steps.  
- Append **only necessary information** to {folder}/metadata.txt to minimize token usage.  
- Use pip-installable names for external libraries. Built-ins should not be listed.
- For image processing, use Python libraries or other gemini model that is working.(no Gemini Vision).  
"""



    chat =await get_chat_session(parse_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message is not None:
        prompt = retry_message
    elif question_text is not None:
        prompt = question_text
    else:
        raise ValueError("parse_question_with_llm: both question_text and retry_message are None")
    
    # Path to the file
    file_path = os.path.join(folder, "metadata.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")



    # Access chat history
    # Example: Save to JSON   

    # Sending response
    response = chat.send_message(prompt)

    # Save chat history after this turn
    history_data = []
    for msg in chat.history:
        history_data.append({
            "role": msg.role,
            "parts": [str(p) for p in msg.parts]
        })
    chat_history_path = os.path.join(folder, "chat_history.json")
    with open(chat_history_path, "w") as f:
        json.dump(history_data, f, indent=2)



    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        raise ValueError(f"Model returned non-JSON: {response.text[:500]}")
    
    