@echo off
chcp 65001 > nul
echo ========================================================
echo INICIANT SISTEMA DE SIMULACIÓ LOGÍSTICA PORCINA
echo ========================================================

:: 1. GENERAR DADES
echo.
echo [PAS 1/4] Generant dades sintètiques (generate_dates.py)...
.venv\Scripts\python.exe generate_dates.py
if %errorlevel% neq 0 goto error

:: 2. CREAR/REINICIALITZAR BASE DE DADES
echo.
echo [PAS 2/4] Inicialitzant Base de Dades (init_db.py)...
.venv\Scripts\python.exe init_db.py
if %errorlevel% neq 0 goto error

:: 3. EXECUTAR SIMULACIÓ PRINCIPAL
echo.
echo [PAS 3/4] Executant Lògica de Simulació (main.py)...
.venv\Scripts\python.exe main.py
if %errorlevel% neq 0 goto error

:: 4. LLANÇAR DASHBOARD
echo.
echo [PAS 4/4] Arrencant Dashboard Interactiu...
echo Prem Ctrl+C en aquesta finestra per aturar el servidor.
echo.
.venv\Scripts\python.exe -m streamlit run app.py

goto end

:error
echo.
echo S'HA PRODUÏT UN ERROR CRÍTIC. REVISA ELS MISSATGES ANTERIORS.
echo.
pause
exit /b %errorlevel%

:end