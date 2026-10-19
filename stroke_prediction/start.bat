@echo off
title MACS-Net Stroke Risk Intelligence Platform
echo ====================================================================
echo   MACS-Net Encephalic Stroke Risk Intelligence Platform Launcher
echo ====================================================================
echo.

:: Ensure Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your PATH. Please install Python.
    pause
    exit /b
)

:: Install/update requirements
echo [1/3] Verifying Python dependencies...
python -m pip install -r backend\requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Dependency check failed. Trying to start anyway...
)

:: Start backend FastAPI server
echo [2/3] Starting MACS-Net Backend API (FastAPI) on port 8000...
cd backend
start "MACS-Net FastAPI Server" /min python -m uvicorn main:app --host 127.0.0.1 --port 8000
cd ..

:: Wait a brief moment for the backend to initialize
echo Waiting for API server to initialize...
timeout /t 3 /nobreak > nul

:: Launch the browser with the frontend SPA
echo [3/3] Launching MACS-Net Web Interface...
start "" "%~dp0index.html"

echo.
echo ====================================================================
echo   MACS-Net is now running in your browser!
echo   - Backend Server:  http://localhost:8000
echo   - API Documentation: http://localhost:8000/docs
echo ====================================================================
echo.
echo Press any key to stop the FastAPI backend and exit.
pause > nul

:: Taskkill the uvicorn process running on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr 8000 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>nul
)
echo Backend server stopped.
exit
