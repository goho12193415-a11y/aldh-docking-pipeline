@echo off
cd /d "%~dp0"
python scripts\gui_app.py
if errorlevel 1 pause
