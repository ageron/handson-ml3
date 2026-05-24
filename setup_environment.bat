@echo off
echo ==========================================
echo   Hands-On ML 3rd Edition Setup
echo ==========================================
echo.

REM Change to project directory
cd /d "k:\learning\technical\courses\Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow\handson-ml3"

echo Step 1: Accepting Terms of Service and creating environment...
echo This will take 10-20 minutes. Please be patient!
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
call conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment may already exist. Trying to update...
    call conda env update -f environment.yml
)

echo.
echo Step 2: Registering Jupyter kernel...
call conda activate homl3
python -m ipykernel install --user --name=python3

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo To start learning, run these commands:
echo.
echo   conda activate homl3
echo   jupyter notebook
echo.
echo Or double-click the start_jupyter.bat file!
echo.
pause
