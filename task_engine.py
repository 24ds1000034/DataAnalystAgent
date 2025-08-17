import subprocess
import sys
import traceback
from typing import List
import datetime
import os
import asyncio
import uuid

import re

try:
    import black  # optional code formatter
except Exception:
    black = None


def _normalize_import_name(pkg: str) -> str:
    base = re.split(r"[<>=!~\[]", pkg.strip(), maxsplit=1)[0].lower()
    mapping = {
        "beautifulsoup4": "bs4",
        "scikit-learn": "sklearn",
        "opencv-python": "cv2",
        "pillow": "PIL",
        "pyyaml": "yaml",
        "python-dotenv": "dotenv",
        "google-generativeai": "google.generativeai",
    }
    return mapping.get(base, base.replace("-", "_"))


async def run_python_code(code: str, libraries: List[str], folder: str = "uploads", python_exec = sys.executable) -> dict:

    # Create a unique work directory per run
    work_dir = os.path.join(folder, f"job_{uuid.uuid4().hex[:8]}")
    os.makedirs(work_dir, exist_ok=True)

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # File where we’ll log execution results
    log_file_path = os.path.join(folder, "execution_result.txt")

    def log_to_file(content: str):
        """Append timestamped content to the log file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{timestamp}]\n{content}\n{'-'*40}\n")

    # Step 1: Check & install required libraries in the selected venv
    for lib in libraries:
        try:
            # check if already installed
            import_name = _normalize_import_name(lib)
            check_cmd = [
                python_exec,
                "-c",
                f"import importlib.util, sys; "
                f"sys.exit(0) if importlib.util.find_spec('{import_name}') else sys.exit(1)"
            ]
            result = await asyncio.to_thread(subprocess.run, check_cmd)
            if result.returncode != 0:  # not installed
                log_to_file(f"📦 Installing {lib} ...")
                await asyncio.to_thread(subprocess.check_call, [python_exec, "-m", "pip", "install", lib, "--upgrade", "--disable-pip-version-check", "--no-input"])
            else:
                log_to_file(f"✅ {lib} already installed.")
        except Exception as install_error:
            error_message = f"❌ Failed to install library '{lib}':\n{install_error}"
            log_to_file(error_message)
            return {"code": 0, "output": error_message}

    # Step 2: Run the code in the isolated venv
    try:
        if black:
            try:
                code_formatted = black.format_str(code, mode=black.Mode())
            except Exception:
                code_formatted = code
        else:
            code_formatted = code

        log_to_file(f"📜 Executing Code:\n{code_formatted}")

        # Save code to job-specific file
        code_file_path = os.path.join(work_dir, "script.py")
        with open(code_file_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code_formatted)
        # Run in subprocess with chosen venv
        result = await asyncio.to_thread(
        subprocess.run,
        [python_exec, code_file_path],
        capture_output=True,
        text=True,
        timeout=600
        )


        if result.returncode == 0:
            success_message = f"✅ Code executed successfully:\n{result.stdout}"
            log_to_file(success_message)
            return {"code": 1, "output": result.stdout}
        else:
            error_message = f"❌ Execution error:\n{result.stderr}"
            log_to_file(error_message)
            return {"code": 0, "output": error_message}

    except Exception as e:
        error_details = f"❌ Error during code execution:\n{traceback.format_exc()}"
        log_to_file(error_details)
        return {"code": 0, "output": error_details}
