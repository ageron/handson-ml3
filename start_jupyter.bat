@echo off
cd /d "k:\learning\technical\courses\Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow\handson-ml3"
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" homl3
start "" jupyter notebook
