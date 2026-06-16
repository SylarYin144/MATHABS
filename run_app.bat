@echo off
setlocal

echo --- Mathabs 1.00 - Ejecucion (Simplificado) ---

set VENV_DIR=matabs_env
set PYTHON_EXE_IN_VENV="%VENV_DIR%\Scripts\python.exe"

REM 1. Verificar si el entorno virtual existe
echo.
echo Verificando entorno virtual .\%VENV_DIR% ...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Entorno virtual .\%VENV_DIR% no encontrado.
    echo Por favor, ejecuta install_and_run_app.bat primero.
    pause
    goto :eof
)
echo Entorno virtual encontrado.

REM 2. Ejecutar aplicacion
echo.
echo Ejecutando Mathabs 1.00 (MATLAB_main_app.py) desde el entorno...
%PYTHON_EXE_IN_VENV% MATLAB_main_app.py

echo.
echo --- Script Finalizado ---
pause

:eof
echo.
echo Saliendo del script.
endlocal
