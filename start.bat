@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  DeepThink AIOS — Windows Start Script
REM  Equivalent of start.sh for Windows environments.
REM ═══════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║           DeepThink AIOS — Windows Launcher                 ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

REM ── Step 1: Find Python ────────────────────────────────────────────
set PYTHON_CMD=
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python3
    ) else (
        where py >nul 2>&1
        if %errorlevel% equ 0 (
            set PYTHON_CMD=py
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Found Python: %PYTHON_CMD%
%PYTHON_CMD% --version

REM ── Step 2: Create and activate virtual environment ────────────────
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM Activate the venv
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

REM ── Step 3: Install Python dependencies ────────────────────────────
echo [SETUP] Installing Python dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Some packages may have failed to install. Continuing anyway...
    echo [WARN] You may need to install Visual Studio Build Tools for some packages.
)
echo [OK] Python dependencies installed.

REM ── Step 4: Check for Node.js ──────────────────────────────────────
set HAS_NODE=0
where node >nul 2>&1
if %errorlevel% equ 0 (
    set HAS_NODE=1
    echo [OK] Found Node.js:
    node --version
) else (
    echo [WARN] Node.js not found. Frontend will not be available.
    echo [WARN] Install Node.js 18+ from https://nodejs.org
)

REM ── Step 5: Install frontend dependencies ──────────────────────────
if %HAS_NODE% equ 1 (
    if exist "frontend\package.json" (
        echo [SETUP] Installing frontend dependencies...
        pushd frontend
        if not exist "node_modules" (
            call npm install --no-audit --no-fund 2>nul
            if %errorlevel% neq 0 (
                echo [WARN] npm install had errors. Frontend may not work.
            )
        )
        popd
        echo [OK] Frontend dependencies ready.
    )
)

REM ── Step 6: Check port availability ────────────────────────────────
set BACKEND_PORT=8000
set FRONTEND_PORT=5173

netstat -ano | findstr ":%BACKEND_PORT% " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARN] Port %BACKEND_PORT% is already in use. Backend may fail to start.
)

netstat -ano | findstr ":%FRONTEND_PORT% " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARN] Port %FRONTEND_PORT% is already in use. Frontend may fail to start.
)

REM ── Step 7: Start Backend ──────────────────────────────────────────
echo.
echo [START] Launching backend on port %BACKEND_PORT%...
start "DeepThink-AIOS Backend" cmd /c "call venv\Scripts\activate.bat && cd backend && python app.py"

REM Give backend a moment to start
timeout /t 3 /nobreak >nul

REM ── Step 8: Start Frontend ─────────────────────────────────────────
if %HAS_NODE% equ 1 (
    if exist "frontend\package.json" (
        echo [START] Launching frontend on port %FRONTEND_PORT%...
        start "DeepThink-AIOS Frontend" cmd /c "cd frontend && npm run dev"
    )
)

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║  DeepThink AIOS is starting!                                ║
echo  ║                                                              ║
echo  ║  Backend:  http://localhost:%BACKEND_PORT%                         ║
echo  ║  Frontend: http://localhost:%FRONTEND_PORT%                         ║
echo  ║                                                              ║
echo  ║  Close the backend/frontend windows to stop the servers.     ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

REM ── Step 9: Wait for user to close ─────────────────────────────────
echo Press any key to stop all servers...
pause >nul

REM ── Step 10: Cleanup — kill child processes ────────────────────────
echo [STOP] Shutting down servers...
taskkill /FI "WINDOWTITLE eq DeepThink-AIOS Backend" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq DeepThink-AIOS Frontend" /F >nul 2>&1

echo [DONE] DeepThink AIOS stopped.
endlocal
