@echo off
setlocal
echo DEBUG: INICIO SCRIPT
pause

echo ==============================================================
echo  Script de Instalacion y Ejecucion para MATABS
echo ==============================================================
echo.
pause

REM --- CONFIGURACION ---
set VENV_NAME=matabs_env
echo DEBUG: VENV_NAME seteado
pause
set REQUIREMENTS_FILE=requirements_matabs.txt
echo DEBUG: REQUIREMENTS_FILE seteado
pause
set MAIN_APP_PYTHON_SCRIPT=MATLAB_main_app.py
echo DEBUG: MAIN_APP_PYTHON_SCRIPT seteado
pause
REM --- FIN CONFIGURACION ---

REM Directorio donde se encuentra este script (asumido como raíz del proyecto)
set "TEMP_SCRIPT_PATH=%~dp0"
echo DEBUG: TEMP_SCRIPT_PATH es "%TEMP_SCRIPT_PATH%"
pause
set "SCRIPT_DIR=%TEMP_SCRIPT_PATH:~0,-1%"
echo DEBUG: SCRIPT_DIR es "%SCRIPT_DIR%"
pause

echo Directorio del Script (Raiz del Proyecto): "%SCRIPT_DIR%"
echo.
pause

REM --- 1. VERIFICAR PYTHON ---
echo Verificando instalacion de Python...
pause
python --version >nul 2>&1
echo DEBUG: Comando python --version ejecutado, Errorlevel es %errorlevel%
pause
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no se encuentra en el PATH.
    echo        Por favor, instale Python (version 3.x recomendada) y asegurese
    echo        de que este anadido al PATH del sistema.
    echo.
    pause
    exit /b 1
)
python --version
echo Python encontrado.
echo.
pause

echo DEBUG: Llegamos al final de la seccion de verificacion de Python sin error aparente en el BAT.
pause

REM (Resto del script original)
echo Verificando archivos necesarios en "%SCRIPT_DIR%"...
if not exist "%SCRIPT_DIR%\%REQUIREMENTS_FILE%" (
    echo ERROR: El archivo de requerimientos "%REQUIREMENTS_FILE%" no se encuentra en "%SCRIPT_DIR%".
    echo.
    pause
    exit /b 1
)
if not exist "%SCRIPT_DIR%\%MAIN_APP_PYTHON_SCRIPT%" (
    echo ERROR: El script principal de la aplicacion "%MAIN_APP_PYTHON_SCRIPT%" no se encuentra en "%SCRIPT_DIR%".
    echo.
    pause
    exit /b 1
)
echo Archivos necesarios encontrados.
echo.

set "VENV_DIR=%SCRIPT_DIR%\%VENV_NAME%"
echo Directorio del entorno virtual: "%VENV_DIR%"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo El entorno virtual "%VENV_NAME%" no parece existir o esta incompleto.
    echo Creando entorno virtual en "%VENV_DIR%"...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual en "%VENV_DIR%".
        echo        Verifique sus permisos y la instalacion de Python.
        echo.
        pause
        exit /b 1
    )
    echo Entorno virtual "%VENV_NAME%" creado exitosamente.
) else (
    echo El entorno virtual "%VENV_NAME%" ya existe en "%VENV_DIR%".
)
echo.

echo Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno virtual.
    echo        Verifique que "%VENV_DIR%\Scripts\activate.bat" existe.
    echo.
    pause
    exit /b 1
)
echo Entorno virtual activado. (Puede ver el prefijo (%VENV_NAME%) en la linea de comandos)
echo.

echo Actualizando pip en el entorno virtual...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ADVERTENCIA: No se pudo actualizar pip. Se continuara con la version actual de pip.
) else (
    echo pip actualizado correctamente.
)
echo.

echo Instalando/verificando dependencias desde "%SCRIPT_DIR%\%REQUIREMENTS_FILE%"...
python -m pip install -r "%SCRIPT_DIR%\%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo ERROR: No se pudieron instalar/verificar las dependencias del archivo "%REQUIREMENTS_FILE%".
    echo        Revise el contenido del archivo y los mensajes de error de pip.
    echo        Si hay errores de compilacion (especialmente para paquetes como numpy, scipy, matplotlib),
    echo        es posible que necesite instalar Microsoft C++ Build Tools:
    echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    pause
    exit /b 1
)
echo Dependencias instaladas/verificadas correctamente.
echo.

echo Ejecutando la aplicacion: "%SCRIPT_DIR%\%MAIN_APP_PYTHON_SCRIPT%"
echo (La aplicacion se iniciara. Cierre la ventana de la aplicacion para finalizar este script.)
echo.

pushd "%SCRIPT_DIR%"
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
