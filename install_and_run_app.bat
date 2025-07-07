@echo off
setlocal
echo ==============================================================
echo  Script de Instalacion y Ejecucion para MATABS
echo ==============================================================
echo.

REM --- CONFIGURACION ---
set VENV_NAME=matabs_env
set REQUIREMENTS_FILE=requirements_matabs.txt
set MAIN_APP_PYTHON_SCRIPT=MATLAB_main_app.py
REM --- FIN CONFIGURACION ---

REM Directorio donde se encuentra este script (asumido como raíz del proyecto)
set "TEMP_SCRIPT_PATH=%~dp0"
set "SCRIPT_DIR=%TEMP_SCRIPT_PATH:~0,-1%"

echo Directorio del Script (Raiz del Proyecto): "%SCRIPT_DIR%"
echo.

REM --- 1. VERIFICAR PYTHON ---
echo Verificando instalacion de Python...
python --version >nul 2>&1
if %errorlevel% NEQ 0 (
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

REM --- 2. VERIFICAR ARCHIVOS NECESARIOS ---
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

REM --- 3. CONFIGURAR Y CREAR ENTORNO VIRTUAL (si no existe) ---
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

REM --- 4. ACTIVAR ENTORNO VIRTUAL ---
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

REM --- 5. ACTUALIZAR PIP (opcional pero recomendado) ---
echo Actualizando pip en el entorno virtual...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ADVERTENCIA: No se pudo actualizar pip. Se continuara con la version actual de pip.
) else (
    echo pip actualizado correctamente.
)
echo.

REM --- 6. INSTALAR DEPENDENCIAS ---
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

REM --- 7. EJECUTAR APLICACION ---
echo Ejecutando la aplicacion: "%SCRIPT_DIR%\%MAIN_APP_PYTHON_SCRIPT%"
echo (La aplicacion se iniciara. Cierre la ventana de la aplicacion para finalizar este script.)
echo.

REM Cambiar al directorio del script para que la app encuentre archivos relativos si los usa
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
