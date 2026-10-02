@echo off
title Industrial Conveyor Vision Analytics - Live Webcam
cd /d "%~dp0"
echo ============================================================
echo   Industrial Conveyor Vision Analytics
echo   Live webcam detection, tracking and counting...
echo ============================================================
python detect_live.py --source 0 --track --count-line
pause
