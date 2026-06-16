@echo off
setlocal

echo ============================================================
echo  Mathabs 1.00 - Ejecutar aplicacion
echo ============================================================
echo.

set VENV_DIR=matabs_env
set PYTHON_EXE="%VENV_DIR%\Scripts\python.exe"
set MAIN_SCRIPT=MATLAB_main_app.py

REM 1. Verificar entorno virtual
echo Verificando entorno virtual .\%VENV_DIR% ...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo ERROR: Entorno virtual no encontrado.
    echo Ejecuta install_and_run_app.bat para instalar el entorno y las dependencias.
    echo.
    pause
    goto :eof
)
echo Entorno virtual encontrado.

REM 2. Verificar que el script principal existe
if not exist "%MAIN_SCRIPT%" (
    echo ERROR: No se encontro %MAIN_SCRIPT%
    echo Asegurate de ejecutar este bat desde la carpeta del proyecto.
    pause
    goto :eof
)

REM 3. Ejecutar aplicacion - capturar errores de inicio
echo.
echo Iniciando Mathabs 1.00...
echo (Cierra esta ventana solo despues de cerrar la aplicacion)
echo.
%PYTHON_EXE% %MAIN_SCRIPT%

if errorlevel 1 (
    echo.
    echo La aplicacion termino con un error (codigo %errorlevel%).
    echo Revisa la consola para mas detalles.
    pause
) else (
    echo.
    echo Aplicacion cerrada correctamente.
    timeout /t 3 >nul
)

:eof
endlocal
