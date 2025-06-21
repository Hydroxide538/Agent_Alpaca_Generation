#!/usr/bin/env python3
"""
Startup script for the CrewAI Workflow Manager
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def check_ollama():
    """Check if Ollama is running"""
    try:
        import aiohttp
        import asyncio
        
        async def check_ollama_status():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                        return response.status == 200
            except:
                return False
        
        is_running = asyncio.run(check_ollama_status())
        if is_running:
            print("✓ Ollama server is running")
        else:
            print("⚠ Ollama server is not running. Please start Ollama if you plan to use Ollama models.")
            print("  You can download Ollama from: https://ollama.ai")
    except ImportError:
        print("⚠ Cannot check Ollama status (aiohttp not installed yet)")

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def start_server(host="0.0.0.0", port=8000, reload=False):
    """Start the FastAPI server"""
    print(f"Starting CrewAI Workflow Manager on {host}:{port}")
    print(f"Frontend will be available at: http://localhost:{port}")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.app:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="CrewAI Workflow Manager Startup Script")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-ollama-check", action="store_true", help="Skip Ollama status check")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CrewAI Workflow Manager - Startup Script")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    
    # Create directories
    create_directories()
    
    # Check Ollama status
    if not args.skip_ollama_check:
        check_ollama()
    
    print("\n" + "=" * 60)
    print("Starting server...")
    print("=" * 60)
    
    # Start server
    start_server(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
