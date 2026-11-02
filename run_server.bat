@echo off
setlocal
cd /d "%~dp0"
set "PORT=8000"
where py >nul 2>nul || (echo Python launcher ^(py^) was not found.& pause & exit /b 1)
py -3 -c "import fastapi,uvicorn,nibabel,cv2,numpy" >nul 2>nul || (echo Required dependencies are missing. Run: py -3 -m pip install -r requirements.txt& pause & exit /b 1)
powershell -NoProfile -Command "if(Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue){exit 1}" >nul 2>nul
if errorlevel 1 (echo Port %PORT% is already in use. Stop the existing server before starting a new one.& pause & exit /b 1)
echo Starting Brain MRI Research and Intelligence Platform...
echo Main application: http://127.0.0.1:%PORT%/
echo Tutorial Lab:      http://127.0.0.1:%PORT%/tutorial-lab
echo MRI studio:        http://127.0.0.1:%PORT%/image-analysis
echo API documentation: http://127.0.0.1:%PORT%/docs
py -3 -m uvicorn backend.main:app --host 127.0.0.1 --port %PORT%
if errorlevel 1 pause
