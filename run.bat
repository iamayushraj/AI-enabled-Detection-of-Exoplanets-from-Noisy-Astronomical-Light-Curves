@echo off
echo ===================================================
echo        Starting ExoplanetAI Hackathon Project
echo ===================================================

echo Starting the FastAPI Backend...
start cmd /k "set PYTHONIOENCODING=utf-8 && set PYTHONPATH=. && python run.py api"

echo Starting the Streamlit Dashboard...
start cmd /k "set PYTHONIOENCODING=utf-8 && set PYTHONPATH=. && python run.py dashboard"

echo.
echo Both services are now booting up in separate terminal windows!
echo.
echo Dashboard UI: http://localhost:8501
echo API Endpoints: http://localhost:8000/docs
echo ===================================================
