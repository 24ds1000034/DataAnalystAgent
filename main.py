from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import logging
from fastapi.responses import HTMLResponse
import difflib
import aiofiles
import time
import itertools
import re

from task_engine import run_python_code
from gemini import parse_question_with_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper funtion to show last 25 words of string s
def last_n_words(s, n=100):
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

def is_csv_empty(csv_path):
    return not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0


# Funtion trip result
def is_base64_image(s: str) -> bool:
    if s.startswith("data:image"):
        return True
    # heuristic: long base64-looking string
    if len(s) > 100 and re.fullmatch(r'[A-Za-z0-9+/=]+', s):
        return True
    return False


def strip_base64_from_json(data: dict) -> dict:
    def _process_value(value):
        if isinstance(value, str) and is_base64_image(value):
            return "[IMAGE_BASE64_STRIPPED]"
        elif isinstance(value, list):
            return [_process_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: _process_value(v) for k, v in value.items()}
        return value

    return _process_value(data)


# Pre-created venv paths (point to the python executable inside each venv)
VENV_PATHS = [
    "venv/bin/python3",
    "venv1/bin/python3",
    "venv2/bin/python3"
]

venv_cycle = itertools.cycle(VENV_PATHS)

@app.post("/api")
async def analyze(request: Request):
    main_loop = 0
    while main_loop < 3:
        try:
            # Create a unique folder for this request
            request_id = str(uuid.uuid4())
            request_folder = os.path.join(UPLOAD_DIR, request_id)
            os.makedirs(request_folder, exist_ok=True)

            # Setting up file for llm response
            llm_response_file_path = os.path.join(request_folder, "llm_response.txt")

            

            # Setup logging for this request
            log_path = os.path.join(request_folder, "app.log")
            logger = logging.getLogger(request_id)
            logger.setLevel(logging.INFO)
            # Remove previous handlers if any (avoid duplicate logs)
            if logger.hasHandlers():
                logger.handlers.clear()
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # Also log to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            logger.info("Step-1: Folder created: %s", request_folder)

            form = await request.form()
            question_text = None
            saved_files = {}

            # Save all uploaded files to the request folder
            for field_name, value in form.items():
                if hasattr(value, "filename") and value.filename:  # It's a file
                    file_path = os.path.join(request_folder, value.filename)
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await value.read())
                    saved_files[field_name] = file_path

                    # If it's questions.txt, read its content
                    if field_name == "question.txt":
                        async with aiofiles.open(file_path, "r") as f:
                            question_text = await f.read()
                else:
                    saved_files[field_name] = value

            # Fallback: If no questions.txt, use the first file as question
            if question_text is None and saved_files:
                target_name = "question.txt"
                file_names = list(saved_files.keys())

                # Find the closest matching filename
                closest_matches = difflib.get_close_matches(target_name, file_names, n=1, cutoff=0.6)
                if closest_matches:
                    selected_file_key = closest_matches[0]
                else:
                    selected_file_key = next(iter(saved_files.keys()))  # fallback to first file

                selected_file_path = saved_files[selected_file_key]

                async with aiofiles.open(selected_file_path, "r") as f:
                    question_text = await f.read()

            python_exec = next(venv_cycle)  # pick next venv
            logger.info("Using Python executable: %s", python_exec)

            user_prompt = f"""
        I know nothing about data analytics. To solve my question, follow this exact process:

        ### Step 1:
        I will give you a question statement with possible data sources (URL, CSV, database, etc.).  
        Your first task: generate code that extracts **basic info** about the data source:
        - URL → Scrape and summarize tables, headings, and important structures. Keep summary under 100 tokens.
        - CSV/Excel → Return first 3 rows.
        - Database → Show table names or schema preview.
        - PDF/Text/Image → Extract a small preview (not full content).

        ### Step 2:
        Download or extract the required data and save it in {request_folder}.

        ### Step 3:
        I will pass you the collected info from {request_folder}/metadata.txt.  
        Then, generate code to solve the question fully and save the **final answer** in {request_folder}/result.txt (or result.json if the format is structured).

        ### Step 4:
        I will give you the content of {request_folder}/result.txt.  
        - If correct → mark `run_this=0`.  
        - If wrong → provide corrected code to recompute the answer.  

        ### Error Handling:
        If I give you an error message:
        - Fix the exact issue and return corrected code.
        - If the error repeats, generate entirely new code.

        ### Output Format:
        You must always answer in **valid JSON** like this:
        {{
            "code": "<python_code_here>",
            "libraries": ["list", "of", "external_libraries"],
            "run_this": 1 or 0
        }}

        ### Additional Rules:
        - Save extracted info into {request_folder}/metadata.txt (append mode).  
        - Save final answers in {request_folder}/result.txt (or {request_folder}/result.json if structured).  
        - Always prepend file access with {request_folder}/filename.  
        - Use only necessary pip-installable external libraries.  

        Example Output: if asked to send a JSON like this:
        ["What is the meadian of data", "What is mean", "provide base64 image"]

        Then, send result like this:

        [20, 25, "base64 image  url here"]
        I mean don't send answers like key:answer, unless it is specified.
        - All the uploaded files will be in the folder {request_folder}. You can access them like this {request_folder}/filename.
        """


            question_text = str("<question>") +  question_text+ "</question>"  + str(user_prompt)
            logger.info("Step-2: File sent %s", saved_files)

            """
            Orchestrates the LLM-driven analytical workflow.
            """
            session_id = request_id
            retry_message = None
            start_time = time.time()

            # Ensure folder exists
            os.makedirs("uploads", exist_ok=True)

            runner = 1

            # Loops to ensure we get a valid json reponse
            max_attempts = 3
            attempt = 0
            response = None
            error_occured = 0
            
            while attempt < max_attempts:
                logger.info("🤖 Step-1: Getting scrap code and metadata from llm. Tries count = %d", attempt)
                try:
                    if error_occured == 0:
                        response = await parse_question_with_llm(
                                    question_text=question_text,
                                    uploaded_files=saved_files,
                                    folder=request_folder,
                                    session_id=session_id,
                                    retry_message=retry_message
                                )
                    else:
                        logger.info("🤖 Step-1: Retrying with error message: %s", retry_message)
                        response = await parse_question_with_llm(retry_message=retry_message, folder=request_folder, session_id=request_id)
                    # Check if response is a valid dict (parsed JSON)
                    if isinstance(response, dict):
                        logger.info("🤖 Step-1: Successfully parsed response from LLM.")
                        break
                except Exception as e:
                    error_occured = 1
                    retry_message = (
            "⚠️ The previous response was not valid JSON.\n"
            "Your task: Fix the issue and return a STRICTLY valid JSON object.\n"
            "Do not include explanations, text, or comments — only JSON.\n\n"
            "Error details (last 100 words):\n<error>"
            + last_n_words(str(e), 100) +
            "</error>\n\n"
            "Expected JSON format:\n"
            "{\n"
            '   "code": "<python_code_here_to_run_in_REPL>",\n'
            '   "libraries": ["list", "of", "external_libraries"],\n'
            '   "run_this": 0 or 1\n'
            "}"
        )

                    logger.error("❌🤖 Step-1: Error in parsing the result. %s", retry_message)
                attempt += 1


            if not isinstance(response, dict):
                logger.error("❌🤖 Step-1: Could not get valid response from LLM after retries.")
                return JSONResponse({"message": "Error_first_llm_call: Could not get valid response from LLM after retries."})

            # Extract code, libraries, and run_this from the response
            code_to_run = response.get("code", "")
            required_libraries = response.get("libraries", [])
            runner = response.get("run_this", 1)

            logger.info("💻 Step-3: Entering loop")
            loop_counter = 0
            while runner == 1:
                loop_counter += 1
                # Check timeout
                #if time.time() - start_time > 500:
                #    print("⏳ Timeout: 150 seconds exceeded.")
                #    break

                logger.info(f"💻 Loop-{loop_counter}: Running LLM code.")
                # Step 2: Run the generated code
                execution_result =await run_python_code(
                    code=code_to_run,
                    libraries=required_libraries,
                    folder=request_folder,
                    python_exec=python_exec
                )

                # Step 3: Check if execution failed
                if execution_result["code"] == 0:
                    logger.error(f"❌💻 Loop-{loop_counter}: Code execution failed: %s", last_n_words(execution_result["output"]))
                    retry_message =str("<error_snippet>") + last_n_words(execution_result["output"]) + str("</error_snippet>") +str("Solve this error or give me new freash code")
                else:
                    logger.info(f"✅💻 Loop-{loop_counter}: Code executed successfully.")
                    # Read metadata
                    metadata_file = os.path.join(request_folder, "metadata.txt")
                    if not os.path.exists(metadata_file):
                        print("❌📁 metadata.txt not found.")
                        continue
                    
                    with open(metadata_file, "r") as f:
                        metadata = f.read()
                    retry_message =str("<metadata>") + metadata + str("</metadata>")
                
                # Checking if result.txt exists
                result_file = os.path.join(request_folder, "result.txt")
                result_path = os.path.join(request_folder, "result.json")

                if os.path.exists(result_path) or os.path.exists(result_file):
                    logger.info(f"✅📁 Loop-{loop_counter}: Found result files.")
                    if os.path.exists(result_path):
                        # Code for reading result.json
                        with open(result_path, "r") as f:
                            result = f.read()
                    elif os.path.exists(result_file):
                        # Code for reading result.txt
                        with open(result_file, "r") as f:
                            result = f.read()

                    result = strip_base64_from_json(result)

                    print("✅ Checking results")
                    # Step 4: Verify the answer with the LLM
                    verification_prompt = f"""
            Check if this answer looks correct:  
            <result> {result} </result>  

            - If you think the result is correct and in correct format→ return JSON with `"run_this": 0`.  
            - If wrong or incomplete → generate **new corrected code** that produces the right result.  
            - Remember this: base64 images are replaced to this text '[IMAGE_BASE64_STRIPPED]' to save token, for images only confirm if this tag is present or not
            - If the question specifies a JSON answer format → save the answer in {request_folder}/result.json.  
            - If some values are missing, fill with placeholder/random values but keep the correct JSON structure.  
            - Only set `"run_this": 1` if the computation must be redone with fresh code.  
            - If it takes more than 3 retries then, just send blank json format with code, libraries and run_this = 0. 

            Always return JSON only.
            """

                    # Loops to ensure we get a valid json reponse
                    max_attempts = 3
                    attempt = 0
                    response = None
                    error_occured = 0

                    while attempt < max_attempts:
                        logger.info(f"🤖 Loop-{loop_counter}: Checking result validity.")
                        try:
                            if error_occured == 0:
                                verification = await parse_question_with_llm(
                                    retry_message=verification_prompt,
                                    uploaded_files=saved_files,
                                    folder=request_folder,
                                    session_id=session_id,
                                    )
                            else:
                                logger.error(f"❌🤖 Loop-{loop_counter}: Invalid json response. %s", retry_message)
                                verification = await parse_question_with_llm(retry_message=retry_message, folder=request_folder, session_id=request_id)
                            # Check if response is a valid dict (parsed JSON)
                            if isinstance(verification, dict):
                                break
                        except Exception as e:
                            error_occured = 1
                            retry_message = (
                "⚠️ The previous response was not valid JSON.\n"
                "Your task: Fix the issue and return a STRICTLY valid JSON object.\n"
                "Do not include explanations, text, or comments — only JSON.\n\n"
                "Error details (last 100 words):\n<error>"
                + last_n_words(str(e), 100) +
                "</error>\n\n"
                "Expected JSON format:\n"
                "{\n"
                '   "code": "<python_code_here_to_run_in_REPL>",\n'
                '   "libraries": ["list", "of", "external_libraries"],\n'
                '   "run_this": 0 or 1\n'
                "}"
            )

                            logger.error(f"❌🤖 Loop-{loop_counter}: Error in parsing the result. %s", retry_message)
                        attempt += 1


                    if not isinstance(verification, dict):
                        logger.error(f"❌🤖 Loop-{loop_counter}: Error: Could not get valid response for validation response.")
                        print(verification)
                        runner = 0

                    if isinstance(verification, dict):
                        code_to_run = verification.get("code", "")
                        required_libraries = verification.get("libraries", [])
                        runner = verification.get("run_this", 0)  # Assume False if not provided
                        if runner == 1:
                            logger.info(f"💻 Loop-{loop_counter}: Re-running code as per validation result.")
                            continue
                        else:
                            logger.info(f"✅ Loop-{loop_counter}: Validation successful, no re-run needed.")
                            
                    try:
                        logger.info(f"💻 Step-6: Running final code.")
                        #Running final code
                        execution_result =await run_python_code(
                            code=code_to_run,
                            libraries=required_libraries,
                            folder=request_folder,
                            python_exec=python_exec
                        )
                        if execution_result["code"] == 0:
                            logger.error(f"❌💻 Step-6: Final code execution failed: %s", last_n_words(execution_result["output"]))
                    except Exception as e:
                        logger.error(f"❌💻 Step-6: Error occurred while running final code: %s", last_n_words(e))

                    # Final step: send the response back by reading the result.txt in JSON format
                    result_path = os.path.join(request_folder, "result.json")

                    if not os.path.exists(result_path):
                        logger.error("❌📁 Step-7: result.json not found. Storing result.txt in .json.")
                        # Checking if result.txt exists
                        result_file = os.path.join(request_folder, "result.txt")
                        
                        # Code for reading result.txt
                        with open(result_file, "r") as f:
                            result = f.read()
                        # Change result.txt content to result.json if possible
                        try:
                            result_path = os.path.join(request_folder, "result.json")
                            with open(result_path, "w") as f:
                                f.write(result)
                        except Exception as e:
                            logger.error(f"❌📁 Step-7: Error occurred while writing result.json: %s", last_n_words(e))

                    
                    with open(result_path, "r") as f:
                        try:
                            logger.info("📁 Step-7: Reading result.json")
                            data = json.load(f)
                            logger.info("✅📁 Step-7: send result back")
                            return JSONResponse(content=data)
                        except Exception as e:
                            logger.error(f"❌📁 Step-7: Error occur while sending result: %s", last_n_words(e))
                            # Return raw content if JSON parsing fails
                            f.seek(0)
                            raw_content = f.read()
                            return JSONResponse({"message": f"Error occured while processing result.json: {e}", "raw_result": raw_content})
            

                # Loops to ensure we get a valid json reponse
                max_attempts = 3
                attempt = 0
                response = None
                error_occured = 0

                while attempt < max_attempts:
                    logger.info(f"🤖 Loop-{loop_counter}: Inside Loop LLM call.")
                    try:
                        if error_occured == 0:
                            response = await parse_question_with_llm(
                            retry_message=retry_message,
                            folder=request_folder,
                            session_id=session_id
                            )
                        else:
                            logger.error(f"❌🤖 Loop-{loop_counter}: Invalid json response. %s", retry_message)
                            response = await parse_question_with_llm(retry_message=retry_message, folder=request_folder, session_id=request_id)
                        # Check if response is a valid dict (parsed JSON)
                        if isinstance(response, dict):
                            break
                    except Exception as e:
                        error_occured = 1
                        retry_message = (
            "⚠️ The previous response was not valid JSON.\n"
            "Your task: Fix the issue and return a STRICTLY valid JSON object.\n"
            "Do not include explanations, text, or comments — only JSON.\n\n"
            "Error details (last 100 words):\n<error>"
            + last_n_words(str(e), 100) +
            "</error>\n\n"
            "Expected JSON format:\n"
            "{\n"
            '   "code": "<python_code_here_to_run_in_REPL>",\n'
            '   "libraries": ["list", "of", "external_libraries"],\n'
            '   "run_this": 0 or 1\n'
            "}"
        )

                        logger.error(f"❌🤖 Loop-{loop_counter}: Error in parsing the result. %s", retry_message)
                    attempt += 1


                if not isinstance(response, dict):
                    logger.error(f"❌🤖 Loop-{loop_counter}: Could not get valid response from LLM after retries.")
                    return JSONResponse({"message": "Error_Inside_loop_call: Could not get valid response from LLM after retries."})

                code_to_run = response.get("code", "")
                required_libraries = response.get("libraries", [])
                runner = response.get("run_this", 1)

                # If somehow it is escaping the loop, then check the result.txt or result.json should not be empty
                if runner == 0:
                    result_path = os.path.join(request_folder, "result.json")

                    if not os.path.exists(result_path):
                        logger.error("❌📁 Step-7: result.json not found. Checking for result.txt.")
                        # Checking if result.txt exists
                        result_file = os.path.join(request_folder, "result.txt")
                        if not os.path.exists(result_file):
                            logger.error("❌📁 result.txt not found. Creating one with blank data.")
                            with open(result_file, "w") as f:
                                f.write('{message:"No result found. Please start over."}')
                            runner = 1
        except Exception as e:
            logger.error(f"❌ Error occurred in main funtion : {e}")
            pass

        main_loop += 1

     

    
