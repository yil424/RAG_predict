@echo on
REM ============================================================
REM  Simple launcher for RAG + ECMO Early-Warning Demo
REM  1) Create virtual environment (.venv) if missing
REM  2) Install dependencies from requirements.txt
REM  3) Run Streamlit app (app_chat.py)
REM ============================================================

setlocal

REM Change to the folder where this script is located
cd /d %~dp0

echo.
echo -----------------------------------------------
echo  RAG + ECMO Early-Warning Demo (Local LLM Mode)
echo -----------------------------------------------
echo.

REM --- Choose your Python command here ---
REM If your system uses "py", change "python" to "py" in the next line.
set PY_CMD=python

REM Show Python version to confirm it works
%PY_CMD% --version
if errorlevel 1 goto NOPYTHON

REM If virtual environment does not exist, create it
if not exist .venv goto MAKEVENV
goto ACTIVATE

:MAKEVENV
echo [INFO] Creating virtual environment (.venv) ...
%PY_CMD% -m venv .venv
if errorlevel 1 goto VENVFAIL

:ACTIVATE
echo [INFO] Activating virtual environment ...
call .venv\Scripts\activate
if errorlevel 1 goto ACTFAIL

echo [INFO] Upgrading pip ...
python -m pip install --upgrade pip

echo [INFO] Installing dependencies from requirements.txt ...
pip install -r requirements.txt

echo.
echo [INFO] Starting Streamlit app (app_chat.py) ...
echo [INFO] Make sure Ollama is installed and "ollama pull llama3.1:8b" has been run.
echo.

streamlit run app_chat.py

goto END

:NOPYTHON
echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
pause
goto END

:VENVFAIL
echo [ERROR] Failed to create virtual environment.
pause
goto END

:ACTFAIL
echo [ERROR] Failed to activate virtual environment.
pause
goto END

:END
echo.
echo [INFO] Done. Press any key to close.
pause

endlocal
