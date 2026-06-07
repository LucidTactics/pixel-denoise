@echo off
REM Same as Denoise.vbs but keeps a console open so errors are visible.
cd /d "%~dp0"
python "%~dp0denoise_gui.py"
echo.
echo ---- GUI closed. If it errored, the message is above. ----
pause
