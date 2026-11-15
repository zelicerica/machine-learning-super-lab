@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
rem Keep this test copy independent of any inherited live-lab override.
set "SML_LAB_PYTHON="
call start-lab.cmd --host 127.0.0.1 --port 8766 --no-browser
