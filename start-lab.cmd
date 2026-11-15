@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if defined SML_LAB_PYTHON (
    "%SML_LAB_PYTHON%" server.py %*
    goto finished
)
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe server.py %*
    goto finished
)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    py -3.12 server.py %*
    goto finished
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    "%LocalAppData%\Programs\Python\Python312\python.exe" server.py %*
    goto finished
)
if exist "%ProgramFiles%\Python312\python.exe" (
    "%ProgramFiles%\Python312\python.exe" server.py %*
    goto finished
)
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>&1
if not errorlevel 1 (
    python server.py %*
    goto finished
)
echo Python 3.12 was not found. In PowerShell or Command Prompt run:
echo.
echo winget install --exact --id Python.Python.3.12 --source winget --scope user
echo.
echo Then close the terminal, reopen it, and double-click this launcher.
echo Read WINDOWS-SETUP.md for custom installation locations and troubleshooting.
pause
exit /b 1
:finished
if errorlevel 1 pause
