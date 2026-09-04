@echo off
title QUANTFLOW - Autonomous Multi-Agent Trading System
color 0A
cls
echo ===============================================================================
echo                 QUANTFLOW: MULTI-AGENT ALGORITHMIC TRADING SYSTEM
echo ===============================================================================
echo [1/3] Checking environment & dependencies...
python -m pip install -q -r requirements.txt
echo [2/3] Starting backend server & multi-agent orchestrator...
start "" http://127.0.0.1:8000
echo [3/3] Dashboard online at http://127.0.0.1:8000
echo.
echo Press CTRL+C to stop the trading server anytime.
echo ===============================================================================
python -m src.main ui
pause
