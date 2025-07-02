@echo off
REM Comprehensive startup script for CrewAI Workflow Manager with GraphRAG (Windows)

echo ================================================================================
echo 🚀 CrewAI Workflow Manager with GraphRAG - Windows Startup
echo ================================================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Docker not found - Neo4j will not be available
    echo Download Docker Desktop from: https://www.docker.com/products/docker-desktop
    set SKIP_NEO4J=--skip-neo4j
) else (
    echo ✅ Docker found
    set SKIP_NEO4J=
)

REM Run the Python startup script
echo.
echo Starting system...
python start_server.py %SKIP_NEO4J% %*

if errorlevel 1 (
    echo.
    echo ❌ Startup failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo System stopped.
pause
