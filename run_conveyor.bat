@echo off
title Industrial Conveyor Vision Analytics - Conveyor Stream
cd /d "%~dp0"
echo ============================================================
echo   Industrial Conveyor Vision Analytics
echo   Running detection on conveyor stream...
echo ============================================================
python detect_live.py --source scratch\test_conveyor_stream.mp4 --track --count-line
pause
