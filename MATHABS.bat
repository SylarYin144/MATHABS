@echo off
setlocal
set e_cmd=exit

REM --- INICIO DE CONFIGURACION ---



REM Nombre del directorio del entorno virtual (se creara dentro de la carpeta del script .bat)
set VENV_NAME=matabs_env

REM Nombres de los archivos clave de la aplicacion
set REQUIREMENTS_FILE_NAME=requirements_matabs.txt
set MAIN_APP_FILE_NAME=matlab_main_app.py
REM --- FIN DE CONFIGURACION ---
REM El directorio raiz de la aplicacion es el directorio donde se encuentra este script .bat
set APP_ROOT_DIR=%~dp0
REM Eliminar la barra invertida final de APP_ROOT_DIR si existe, para consistencia.
if "%%APP_ROOT_DIR:~-1%%"=="\" set APP_ROOT_DIR=%%APP_ROOT_DIR:~0,-1%%

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no se encuentra en el PATH.
    echo Por favor, instala Python y asegurate de que este en el PATH.
    pause
    exit /b 1
)
echo Python encontrado.
REM --- VERIFICAR ARCHIVOS NECESARIOS EN EL DIRECTORIO DEL SCRIPT ---
echo Buscando archivos de la aplicacion en: "%APP_ROOT_DIR%"
if not exist "%APP_ROOT_DIR%\%REQUIREMENTS_FILE_NAME%" (
    echo ERROR: Archivo de requerimientos "%REQUIREMENTS_FILE_NAME%" no encontrado en "%APP_ROOT_DIR%".
    pause
    %e_cmd% /b 1
)
if not exist "%APP_ROOT_DIR%\%MAIN_APP_FILE_NAME%" (
    echo ERROR: Archivo principal de la aplicacion "%MAIN_APP_FILE_NAME%" no encontrado en "%APP_ROOT_DIR%".
    pause
    %e_cmd% /b 1
)
echo Archivos de la aplicacion encontrados.
echo Directorio raiz de la aplicacion (MATABS) establecido en: "%APP_ROOT_DIR%"
echo.

REM --- CONFIGURACION DEL ENTORNO VIRTUAL ---
set VENV_DIR=%APP_ROOT_DIR%\%VENV_NAME%
echo Directorio del entorno virtual: "%VENV_DIR%"

if not exist "%VENV_DIR%" (
    echo Creando entorno virtual en "%VENV_DIR%"...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual en "%VENV_DIR%".
        pause
        exit /b 1
    )
    echo Entorno virtual creado.
) else (
    echo El entorno virtual "%VENV_DIR%" ya existe.
)
echo.

echo Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno virtual.
    echo Verifica que "%VENV_DIR%\Scripts\activate.bat" existe y es ejecutable.
    pause
    exit /b 1
)
echo Entorno virtual activado.
echo.

set REQUIREMENTS_PATH=%APP_ROOT_DIR%\%REQUIREMENTS_FILE_NAME%
echo Instalando dependencias desde "%REQUIREMENTS_PATH%"...
pip install -r "%REQUIREMENTS_PATH%"
if errorlevel 1 (
    echo ERROR: No se pudieron instalar las dependencias. Revisa el archivo "%REQUIREMENTS_PATH%" y la salida de pip.
    pause
    exit /b 1
)
echo Dependencias instaladas correctamente.
echo.

set MAIN_APP_SCRIPT_PATH_IN_ROOT=%MAIN_APP_FILE_NAME%
echo Ejecutando la aplicacion: "%APP_ROOT_DIR%\%MAIN_APP_SCRIPT_PATH_IN_ROOT%"
echo.

REM Cambiamos al directorio de la aplicacion para que las rutas relativas dentro del script de Python funcionen correctamente.
pushd "%APP_ROOT_DIR%"
python "%MAIN_APP_SCRIPT_PATH_IN_ROOT%"
popd

echo.
echo La aplicacion ha finalizado.
pause

endlocal
exit /b 0