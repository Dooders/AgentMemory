@echo off
setlocal enabledelayedexpansion

REM Set Python path to include parent directory
set PYTHONPATH=..;%PYTHONPATH%

REM Configuration
set BASE_THREAD_COUNT=5
set BASE_OPERATIONS=1000
set OUTPUT_FILE=concurrent_results.csv

REM Create empty output file if it doesn't exist
if not exist %OUTPUT_FILE% type nul > %OUTPUT_FILE%

REM Test with increasing thread counts
echo Running concurrent access tests with increasing thread counts...
for /l %%x in (1, 1, 5) do (
    set /a THREADS=!BASE_THREAD_COUNT! * %%x
    echo Testing with !THREADS! concurrent threads...
    python concurrent_test.py --threads !THREADS! --operations !BASE_OPERATIONS! --output !OUTPUT_FILE!
    timeout /t 5 /nobreak > nul
)

REM Test with different write/read ratios
echo.
echo Running concurrent access tests with different write/read ratios...
set /a THREADS=!BASE_THREAD_COUNT! * 2
for %%r in (0.25 0.5 0.75 0.9) do (
    echo Testing with write ratio %%r...
    python concurrent_test.py --threads !THREADS! --operations !BASE_OPERATIONS! --write-ratio %%r --output !OUTPUT_FILE!
    timeout /t 5 /nobreak > nul
)

REM Test with high load
echo.
echo Running high load concurrent test...
set /a HIGH_THREADS=!BASE_THREAD_COUNT! * 10
set /a HIGH_OPS=!BASE_OPERATIONS! / 2
python concurrent_test.py --threads !HIGH_THREADS! --operations !HIGH_OPS! --output !OUTPUT_FILE!

REM Race condition test
echo.
echo Running race condition test...
python concurrent_test.py --threads 100 --operations 50 --output !OUTPUT_FILE!

echo.
echo All concurrent tests completed. Results saved to !OUTPUT_FILE!

endlocal
pause 