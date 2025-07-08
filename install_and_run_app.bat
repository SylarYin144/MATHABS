@echo off
setlocal

echo --- MATABS Instalacion y Ejecucion (Simplificado) ---

set VENV_DIR=matabs_env
set PYTHON_EXE_IN_VENV="%VENV_DIR%\Scripts\python.exe"
set PIP_EXE_IN_VENV="%VENV_DIR%\Scripts\pip.exe"

REM 1. Verificar Python en el PATH (necesario para crear venv)
echo.
echo Verificando Python en el PATH...
python --version
if errorlevel 1 (
    echo ERROR: Python no esta en el PATH. Por favor, instalalo y anadelo al PATH.
    pause
    goto :eof
)
echo Python en el PATH encontrado.

REM 2. Crear entorno virtual si no existe
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo Creando entorno virtual en .\%VENV_DIR% ...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR al crear el entorno virtual.
        pause
        goto :eof
    )
    echo Entorno virtual creado.
) else (
    echo.
    echo Entorno virtual .\%VENV_DIR% ya existe.
)

REM 3. Activar (implícito por llamar a ejecutables del venv) e instalar dependencias
echo.
echo Instalando dependencias (esto puede tardar)...
%PIP_EXE_IN_VENV% install -r requirements_matabs.txt
if errorlevel 1 (
    echo ERROR al instalar dependencias. Verifique requirements_matabs.txt y los mensajes de pip.
    echo Si ve errores de compilacion, podria necesitar Microsoft C++ Build Tools.
    pause
    goto :eof
)
echo Dependencias instaladas/verificadas.

REM 4. Ejecutar aplicacion
echo.
echo Ejecutando MATLAB_main_app.py...
%PYTHON_EXE_IN_VENV% MATLAB_main_app.py

echo.
echo --- Script Finalizado ---
pause

:eof
echo.
echo Saliendo del script.
endlocal
