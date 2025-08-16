# task_engine.py

import subprocess
import sys
import traceback
from typing import List
import datetime
import os
import re
import black
import shutil
import uuid


def _resolve_python_exec(python_exec: str | None) -> str:
    """
    Return a valid, absolute path to a Python executable or raise a clear error.
    Resolution order:
      1) explicit python_exec if it exists
      2) common venv paths
      3) sys.executable
      4) shutil.which("python")
    """
    candidates = []
    if python_exec:
        candidates.append(python_exec)

    # Helpful defaults
    for p in (".venv/Scripts/python.exe", ".venv/bin/python3",
              "venv/Scripts/python.exe", "venv/bin/python3"):
        if p not in candidates:
            candidates.append(p)

    if sys.executable and sys.executable not in candidates:
        candidates.append(sys.executable)
    which_py = shutil.which("python")
    if which_py and which_py not in candidates:
        candidates.append(which_py)

    for cand in candidates:
        cand_norm = os.path.normpath(os.path.expandvars(os.path.expanduser(cand)))
        if os.path.isfile(cand_norm):
            return cand_norm

    raise FileNotFoundError(
        "No valid Python interpreter found. Tried: "
        + ", ".join([str(c) for c in candidates if c])
        + ". Ensure Python is installed and/or provide a correct absolute path."
    )


def _normalize_import_name(pkg_name: str) -> str:
    """
    Map pip package names to import names; strip pins/extras.
    """
    base = re.split(r"[<>=!~\[]", pkg_name.strip(), maxsplit=1)[0].lower()
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


async def run_python_code(
    code: str,
    libraries: List[str],
    folder: str = "uploads",
    python_exec: str | None = None,
) -> dict:
    # Resolve interpreter
    try:
        python_path = _resolve_python_exec(python_exec)
    except Exception as e:
        return {"code": 0, "output": f"❌ Could not resolve Python interpreter:\n{e}"}

    # Paths
    os.makedirs(folder, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]
    work_dir = os.path.join(folder, f"job_{job_id}")
    os.makedirs(work_dir, exist_ok=True)

    # Run scripts from the project root so relative paths like "uploads/<uuid>/file"
    # resolve correctly (your generated code uses those).
    project_root = os.path.abspath(os.getcwd())

    log_file_path = os.path.join(folder, "execution_result.txt")

    def log_to_file(content: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{ts}]\n{content}\n{'-'*40}\n")

    # Step 1: install libraries (dedup + mapping)
    uniq_libs = list(dict.fromkeys(libraries or []))
    pip_env = os.environ.copy()
    pip_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    pip_env.setdefault("PYTHONIOENCODING", "utf-8")

    for lib in uniq_libs:
        try:
            import_name = _normalize_import_name(lib)
            check_cmd = [
                python_path, "-c",
                ("import importlib.util, sys; "
                 f"sys.exit(0) if importlib.util.find_spec('{import_name}') else sys.exit(1)")
            ]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_to_file(f"📦 Installing {lib} ...")
                try:
                    install = subprocess.run(
                        [python_path, "-m", "pip", "install", lib, "--upgrade",
                         "--disable-pip-version-check", "--no-input"],
                        capture_output=True,
                        text=True,
                        cwd=work_dir,
                        env=pip_env,
                        timeout=300,
                    )
                except subprocess.TimeoutExpired:
                    msg = f"❌ pip install timed out for '{lib}'."
                    log_to_file(msg)
                    return {"code": 0, "output": msg}

                if install.returncode != 0:
                    msg = (f"❌ Failed to install library '{lib}'.\n"
                           f"stdout:\n{install.stdout}\n\nstderr:\n{install.stderr}")
                    log_to_file(msg)
                    return {"code": 0, "output": msg}
                log_to_file(f"✅ Installed {lib}.")
            else:
                log_to_file(f"✅ {lib} already installed.")
        except FileNotFoundError as fnf:
            msg = (f"❌ Failed to install library '{lib}': Python not found.\n{fnf}\n"
                   f"Interpreter tried: {python_path}")
            log_to_file(msg)
            return {"code": 0, "output": msg}
        except Exception as install_error:
            msg = f"❌ Failed to install library '{lib}':\n{install_error}"
            log_to_file(msg)
            return {"code": 0, "output": msg}

    # Step 2: write and run
    try:
        try:
            code_formatted = black.format_str(code, mode=black.Mode())
        except Exception:
            code_formatted = code

        log_to_file(
            f"📜 Executing Code:\n"
            f"(cwd for script execution): {project_root}\n"
            f"(script file location): {work_dir}\n"
            f"{code_formatted}"
        )

        code_file_path = os.path.abspath(os.path.join(work_dir, "script.py"))
        with open(code_file_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(code_formatted)

        # ✅ Run from project root so 'uploads/<uuid>/...' paths are correct
        result = subprocess.run(
            [python_path, code_file_path],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode == 0:
            log_to_file(f"✅ Code executed successfully:\n{result.stdout}")
            return {"code": 1, "output": result.stdout}
        else:
            error_message = (
                "❌ Execution error:\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )
            log_to_file(error_message)
            return {"code": 0, "output": error_message}

    except FileNotFoundError as fnf:
        error_details = ("❌ Error during code execution (Python not found):\n"
                         f"{fnf}\nInterpreter tried: {python_path}")
        log_to_file(error_details)
        return {"code": 0, "output": error_details}
    except Exception:
        error_details = f"❌ Error during code execution:\n{traceback.format_exc()}"
        log_to_file(error_details)
        return {"code": 0, "output": error_details}
