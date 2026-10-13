@echo off
cd /d "%~dp0"
echo ============================================
echo Repack for release: sync portable training assets, then rebuild EXE
echo ============================================
if exist "release\models" (
  echo Syncing models from the current portable release...
  robocopy "release\models" "models" *.pt /XO >nul
)
if exist "release\data\writers" (
  echo Syncing writers data from packaged-app data folder...
  robocopy "release\data\writers" "data\writers" /E /XO >nul
)
call build_exe.bat
