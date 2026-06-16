@echo off
setlocal

echo ============================================================
echo  Mathabs 1.00 - Compilacion EXE con PyInstaller
echo ============================================================
echo.

set VENV_DIR=matabs_env
set PYTHON_EXE="%VENV_DIR%\Scripts\python.exe"
set PIP_EXE="%VENV_DIR%\Scripts\pip.exe"
set SPEC_FILE=mathabs.spec

REM 1. Verificar entorno virtual
echo Verificando entorno virtual .\%VENV_DIR% ...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Entorno virtual no encontrado.
    echo Ejecuta install_and_run_app.bat primero para crear el entorno.
    pause
    goto :eof
)
echo Entorno virtual encontrado.

REM 2. Verificar / instalar PyInstaller
echo.
echo Verificando PyInstaller...
%PYTHON_EXE% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller no encontrado. Instalando...
    %PIP_EXE% install pyinstaller
    if errorlevel 1 (
        echo ERROR al instalar PyInstaller.
        pause
        goto :eof
    )
)
echo PyInstaller disponible.

REM 3. Compilar con el spec file
echo.
if exist "%SPEC_FILE%" (
    echo Compilando con %SPEC_FILE% ...
    %PYTHON_EXE% -m PyInstaller %SPEC_FILE% --clean --noconfirm
) else (
    echo Archivo %SPEC_FILE% no encontrado. Generando EXE directamente...
    %PYTHON_EXE% -m PyInstaller MATLAB_main_app.py --name mathabs --onefile ^
        --windowed --noconfirm --clean
)

if errorlevel 1 (
    echo.
    echo ERROR en la compilacion. Revisa los mensajes anteriores.
    pause
    goto :eof
)

echo.
echo ============================================================
echo  Compilacion completada. El EXE esta en la carpeta dist\
echo ============================================================
pause

:eof
endlocal
