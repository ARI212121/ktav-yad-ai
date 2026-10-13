@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo Building portable single-file EXE
echo User data will be stored beside the EXE in data\ and models\
echo ============================================
set "KTAVYADAI_EMBEDDED_BROWSER=1"
echo Building with an embedded application window; no external browser is required.
if not exist "icon.ico" (
  echo ERROR: Required application icon is missing: icon.ico.
  exit /b 1
)

for %%F in ("data\corpus_lite.txt" "data\config.json" "data\fixes_general.tsv" "data\.lm_counts.pkl" "data\.lm_bigrams.pkl" "data\.lm_trigrams.pkl" "models\base_synth.pt" "models\general.pt" "models\typed.pt") do (
  if not exist "%%~F" (
    echo ERROR: Required bundled asset is missing: %%~F.
    exit /b 1
  )
)

if not exist "venv\Scripts\python.exe" (
  set "PYTHON_CMD="
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.13 --version >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.13"
    if not defined PYTHON_CMD (
      py -3.14 --version >nul 2>nul
      if not errorlevel 1 set "PYTHON_CMD=py -3.14"
    )
  )
  if not defined PYTHON_CMD (
    where python >nul 2>nul
    if errorlevel 1 goto :python_missing
    python -c "import sys; raise SystemExit(0 if (3,13) <= sys.version_info[:2] < (3,15) else 1)" >nul 2>nul
    if errorlevel 1 goto :python_version
    set "PYTHON_CMD=python"
  )
  echo Creating project virtual environment...
  %PYTHON_CMD% -m venv venv
  if errorlevel 1 goto :venv_error
)

echo Installing or verifying required Python packages...
venv\Scripts\python.exe -m ensurepip --upgrade
if errorlevel 1 goto :dependency_error
venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :dependency_error

if "%KTAVYADAI_EMBEDDED_BROWSER%"=="1" (
  echo Installing embedded application-window packages...
  venv\Scripts\python.exe -m pip install --disable-pip-version-check "pywebview[qt]"
  if errorlevel 1 goto :dependency_error
)

if not exist "venv\Scripts\pyinstaller.exe" (
  echo ERROR: PyInstaller was not installed from requirements.txt.
  exit /b 1
)

venv\Scripts\pyinstaller.exe --noconfirm --clean --distpath release --workpath build KtavYadAI.spec
if errorlevel 1 exit /b 1
if not exist "release\KtavYadAI.exe" (
  echo ERROR: PyInstaller reported success but release\KtavYadAI.exe was not created.
  exit /b 1
)

if not exist "portable" mkdir "portable"
echo Building the standalone OCRHE command-line executable...
venv\Scripts\pyinstaller.exe --noconfirm --clean --distpath portable --workpath build\ocrhe OCRHE.spec
if errorlevel 1 exit /b 1
if not exist "portable\OCRHE.exe" (
  echo ERROR: PyInstaller reported success but portable\OCRHE.exe was not created.
  exit /b 1
)
copy /Y "release\KtavYadAI.exe" "portable\KtavYadAI.exe" >nul
if errorlevel 1 (
  echo ERROR: Could not copy the GUI executable into portable.
  exit /b 1
)
if not exist "portable\data" if exist "release\data" (
  xcopy "release\data" "portable\data" /E /I /H /Y >nul
  if errorlevel 1 exit /b 1
)
if not exist "portable\models" if exist "release\models" (
  xcopy "release\models" "portable\models" /E /I /H /Y >nul
  if errorlevel 1 exit /b 1
)

echo.
echo Done:
echo   release\KtavYadAI.exe       Official GUI release build
echo   portable\KtavYadAI.exe      Portable GUI executable
echo   portable\OCRHE.exe          Portable command-line executable
echo Both portable executables share portable\data and portable\models.
portable\OCRHE.exe --help >nul
if errorlevel 1 (
  echo ERROR: OCRHE command-line help smoke test failed.
  exit /b 1
)
echo Running startup diagnostic with a log...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$exe = (Resolve-Path 'portable\KtavYadAI.exe').Path; $timer = [Diagnostics.Stopwatch]::StartNew(); $p = Start-Process -FilePath $exe -ArgumentList '--diagnose-startup' -WorkingDirectory (Get-Location).Path -Wait -PassThru; $timer.Stop(); Add-Content -Path 'portable\data\startup.log' -Value ('build_process_total_seconds={0:N2}' -f $timer.Elapsed.TotalSeconds); exit $p.ExitCode"
if errorlevel 1 (
  echo ERROR: Startup diagnostic failed. See portable\data\startup.log
  exit /b 1
)
echo Startup diagnostic passed. Log: portable\data\startup.log
exit /b 0

:python_missing
echo ERROR: Install Python 3.13 or 3.14 and enable the Python launcher or PATH.
exit /b 1

:python_version
echo ERROR: Python 3.13 or 3.14 is required to build this application.
exit /b 1

:venv_error
echo ERROR: Could not create the project virtual environment.
exit /b 1

:dependency_error
echo ERROR: Dependency installation failed. Check your internet connection and retry.
exit /b 1
