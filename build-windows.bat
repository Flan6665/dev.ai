@echo off
chcp 65001 >nul
setlocal
rem Сборка Ишки с CUDA на Windows. Сам находит Visual Studio и подключает компилятор C++ (cl.exe),
rem поэтому запускать можно из обычного PowerShell:  .\build-windows.bat 89
rem Число — архитектура видеокарты: RTX 20xx = 75, 30xx = 86, 40xx = 89, 50xx = 120.

set "CAP=%~1"
if "%CAP%"=="" set "CAP=89"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto novs
set "VSDIR="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto novs

echo [1/3] Подключаю компилятор C++ из: %VSDIR%
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl >nul 2>nul
if errorlevel 1 goto novs

echo [2/3] Проверяю CUDA...
where nvcc >nul 2>nul
if errorlevel 1 goto nocuda
nvcc --version | findstr /i "release"

echo [3/3] Собираю (первый раз 10-30 минут), CUDA_COMPUTE_CAP=%CAP%
set "CUDA_COMPUTE_CAP=%CAP%"
cargo build --release --features cuda
if errorlevel 1 goto failed

echo.
echo ГОТОВО! Программа: target\release\ishka.exe
echo В PowerShell запускай так:  .\target\release\ishka.exe --help
exit /b 0

:novs
echo.
echo ОШИБКА: не найден компилятор C++ из Visual Studio.
echo Установи "Build Tools for Visual Studio" (visualstudio.microsoft.com/downloads, внизу страницы)
echo и отметь "Разработка классических приложений на C++". Потом запусти этот скрипт снова.
exit /b 1

:nocuda
echo.
echo ОШИБКА: не найден nvcc. Установи CUDA Toolkit и перезагрузи компьютер.
exit /b 1

:failed
echo.
echo Сборка не удалась. Скопируй последние 30-40 строк выше и пришли их.
exit /b 1
