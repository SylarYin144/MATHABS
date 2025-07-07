@echo off
setlocal
echo ==============================================================
echo  Script de Ejecucion para MATABS (Entorno Existente)
echo ==============================================================
echo.

REM --- CONFIGURACION (debe coincidir con install_and_run_app.bat) ---
set VENV_NAME=matabs_env
set MAIN_APP_PYTHON_SCRIPT=MATLAB_main_app.py
REM --- FIN CONFIGURACION ---

REM Directorio donde se encuentra este script (asumido como raíz del proyecto)
set "TEMP_SCRIPT_PATH=%~dp0"
set "SCRIPT_DIR=%TEMP_SCRIPT_PATH:~0,-1%"

echo Directorio del Script (Raiz del Proyecto): "%SCRIPT_DIR%"
echo.

REM --- 1. VERIFICAR PYTHON (general, no necesariamente el del venv aun) ---
echo Verificando instalacion de Python base...
python --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo ERROR: Python no esta instalado o no se encuentra en el PATH.
    echo        Este script requiere Python para activar el entorno virtual.
    echo.
    pause
    exit /b 1
)
echo Python base encontrado.
echo.

REM --- 2. VERIFICAR ENTORNO VIRTUAL Y ARCHIVO PRINCIPAL ---
set "VENV_DIR=%SCRIPT_DIR%\%VENV_NAME%"
echo Verificando entorno virtual en: "%VENV_DIR%"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: El entorno virtual "%VENV_NAME%" no se encuentra o esta incompleto en "%VENV_DIR%".
    echo        Por favor, ejecute primero el script 'install_and_run_app.bat'
    echo        para crear el entorno e instalar las dependencias.
    echo.
    pause
    exit /b 1
)
echo Entorno virtual "%VENV_NAME%" encontrado.
echo.

if not exist "%SCRIPT_DIR%\%MAIN_APP_PYTHON_SCRIPT%" (
    echo ERROR: El script principal de la aplicacion "%MAIN_APP_PYTHON_SCRIPT%" no se encuentra en "%SCRIPT_DIR%".
    echo.
    pause
    exit /b 1
)
echo Script principal de la aplicacion encontrado.
echo.

REM --- 3. ACTIVAR ENTORNO VIRTUAL ---
echo Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno virtual.
    echo        Verifique que "%VENV_DIR%\Scripts\activate.bat" existe.
    echo.
    pause
    exit /b 1
)
echo Entorno virtual activado.
echo.

REM --- 4. EJECUTAR APLICACION ---
echo Ejecutando la aplicacion: "%SCRIPT_DIR%\%MAIN_APP_PYTHON_SCRIPT%"
echo (La aplicacion se iniciara. Cierre la ventana de la aplicacion para finalizar este script.)
echo.

REM Cambiar al directorio del script para que la app encuentre archivos relativos
pushd "%SCRIPT_DIR%"

REM Ejecutar la aplicación Python. La consola esperará a que la app termine.
call python "%MAIN_APP_PYTHON_SCRIPT%"
echo.
echo Script de Python ha terminado o fallado. Presione una tecla para continuar...
pause

popd
echo.
echo ==============================================================
echo  La aplicacion Python ha finalizado o ha sido cerrada.
echo ==============================================================
echo.
pause
endlocal
exit /b 0
