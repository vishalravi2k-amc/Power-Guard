@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: install Python 3.12, then allow package downloads.
  py -3.12 -m venv .venv
  if errorlevel 1 goto error
)
if not exist .venv\power_guard_ready (
  .venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto error
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto error
  echo ready>.venv\power_guard_ready
)
.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
if errorlevel 1 goto error
exit /b 0
:error
echo Setup or launch failed. Read README.md troubleshooting and the error above.
pause
exit /b 1
