@echo off
set PYTHONPATH=..;%PYTHONPATH%
python concurrent_test.py --threads 5 --operations 100
pause 