@echo off
setlocal

set LOCAL_FLAG=
if not defined ANTHROPIC_API_KEY (
  echo ANTHROPIC_API_KEY is not set. Running local-only report mode.
  set LOCAL_FLAG=--local-only
)

set PY=python
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 (
  if exist "venv_tt\Scripts\python.exe" (
    set PY=venv_tt\Scripts\python.exe
  )
)

if "%LOCAL_FLAG%"=="--local-only" (
  %PY% -c "import docx" >nul 2>&1
  if errorlevel 1 (
    echo Installing missing package into selected Python environment...
    %PY% -m pip install python-docx
    if errorlevel 1 (
      echo Failed to install required package ^(python-docx^).
      pause
      exit /b 1
    )
  )
) else (
  %PY% -c "import anthropic, docx" >nul 2>&1
  if errorlevel 1 (
    echo Installing missing packages into selected Python environment...
    %PY% -m pip install anthropic python-docx
    if errorlevel 1 (
      echo Failed to install required packages ^(anthropic, python-docx^).
      pause
      exit /b 1
    )
  )
)

%PY% generate_training_report.py %LOCAL_FLAG% %*
if errorlevel 1 (
  echo.
  echo Report generation failed.
  pause
  exit /b 1
)

echo.
echo Report generation completed.
pause
