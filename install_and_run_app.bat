@echo off
setlocal

echo ============================================================
echo  Mathabs 1.00 - Instalacion y Ejecucion
echo ============================================================
echo.

set VENV_DIR=matabs_env
set PYTHON_EXE_IN_VENV="%VENV_DIR%\Scripts\python.exe"
set PIP_EXE_IN_VENV="%VENV_DIR%\Scripts\pip.exe"

REM 1. Verificar Python 3.11 en el PATH usando el launcher py
REM    Dado que la aplicacion tiene dependencias muy especificas y estrictas,
REM    se requiere obligatoriamente Python 3.11.
echo Verificando Python 3.11 (py -3.11)...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: No se encontro Python 3.11 en el sistema.
    echo Esta aplicacion tiene requerimientos especificos de librerias antiguas
    echo que requieren obligatoriamente usar Python 3.11.
    echo.
    echo Por favor, descarga e instala Python 3.11 desde:
    echo https://www.python.org/downloads/release/python-3119/
    echo Asegurate de marcar la opcion "Add Python to PATH" durante la instalacion.
    echo.
    pause
    goto :eof
)
echo Python 3.11 encontrado.

REM 2. Crear o verificar entorno virtual
set RECREATE_VENV=0
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo Entorno virtual .\%VENV_DIR% ya existe. Verificando version de Python...
    %PYTHON_EXE_IN_VENV% -c "import sys; sys.exit(0 if list(sys.version_info[0:2]) == [3, 11] else 1)" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ADVERTENCIA: El entorno virtual actual no esta usando Python 3.11.
        echo Se borrara y se volvera a crear con Python 3.11 para evitar conflictos de version.
        echo.
        rmdir /s /q %VENV_DIR%
        set RECREATE_VENV=1
    )
) else (
    set RECREATE_VENV=1
)

if "%RECREATE_VENV%"=="1" (
    echo Creando entorno virtual en .\%VENV_DIR% con Python 3.11...
    py -3.11 -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR al crear el entorno virtual con Python 3.11.
        pause
        goto :eof
    )
    echo Entorno virtual creado.
)

REM 3. Instalar dependencias
echo.
echo Instalando dependencias desde requirements_matabs.txt (esto puede tardar)...
%PIP_EXE_IN_VENV% install -r requirements_matabs.txt
if errorlevel 1 (
    echo.
    echo ERROR al instalar dependencias.
    echo Verifique requirements_matabs.txt y los mensajes de pip.
    echo Si ve errores de compilacion, podria necesitar Microsoft C++ Build Tools.
    pause
    goto :eof
)
echo Dependencias instaladas/verificadas.

REM 4. Ejecutar aplicacion
echo.
echo ============================================================
echo  Iniciando Mathabs 1.00...
echo ============================================================
echo.
%PYTHON_EXE_IN_VENV% MATLAB_main_app.py

echo.
echo --- Aplicacion cerrada. Script finalizado. ---
pause

:eof
echo.
echo Saliendo del script.
endlocal
