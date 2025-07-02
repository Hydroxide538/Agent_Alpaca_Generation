#!/usr/bin/env python3
"""
Comprehensive Startup Script for CrewAI Workflow Manager with GraphRAG
Handles complete system initialization including Neo4j, dependencies, models, and services
"""

import os
import sys
import subprocess
import argparse
import time
import json
import signal
import atexit
from pathlib import Path
import asyncio
import threading
from datetime import datetime

# Set a placeholder OpenAI API key if not already set
if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = 'placeholder_key_not_used'

class SystemManager:
    """Manages all system components for the CrewAI Workflow Manager"""
    
    def __init__(self):
        self.processes = []
        self.neo4j_container = None
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup all processes and containers"""
        print("Cleaning up processes...")
        
        # Stop FastAPI server
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        # Stop Neo4j container if we started it
        if self.neo4j_container:
            try:
                subprocess.run(["docker", "stop", "graphrag_neo4j"], 
                             capture_output=True, timeout=10)
                print("✓ Neo4j container stopped")
            except:
                pass
    
    def print_header(self):
        """Print startup header"""
        print("=" * 80)
        print("🚀 CrewAI Workflow Manager with GraphRAG - Complete Startup")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("1️⃣  Checking Python version...")
        if sys.version_info < (3, 8):
            print("❌ Error: Python 3.8 or higher is required")
            sys.exit(1)
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    
    def check_docker(self):
        """Check if Docker is available"""
        print("\n2️⃣  Checking Docker availability...")
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
                return True
            else:
                print("⚠️  Docker not found - Neo4j will not be available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  Docker not found - Neo4j will not be available")
            return False
    
    def install_dependencies(self, skip_deps=False):
        """Install required dependencies"""
        if skip_deps:
            print("\n3️⃣  Skipping dependency installation...")
            return
        
        print("\n3️⃣  Installing dependencies...")
        try:
            # Check if requirements file exists
            if not os.path.exists("backend/requirements.txt"):
                print("❌ backend/requirements.txt not found")
                sys.exit(1)
            
            # Install dependencies
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            print("Try running: pip install -r backend/requirements.txt")
            sys.exit(1)
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n4️⃣  Creating directories...")
        directories = [
            "uploads", "results", "logs", "vector_db", 
            "backend/uploads", "backend/vector_db", "backend/results"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print("✅ All directories created")
    
    def start_neo4j(self, docker_available=True):
        """Start Neo4j database"""
        if not docker_available:
            print("\n5️⃣  Skipping Neo4j (Docker not available)...")
            return False
        
        print("\n5️⃣  Starting Neo4j database...")
        
        # Check if Neo4j is already running
        try:
            result = subprocess.run(["docker", "ps", "--filter", "name=graphrag_neo4j", "--format", "{{.Names}}"],
                                  capture_output=True, text=True, timeout=10)
            if "graphrag_neo4j" in result.stdout:
                print("✅ Neo4j container already running")
                return True
        except:
            pass
        
        # Start Neo4j using docker-compose
        try:
            if os.path.exists("docker-compose.yml"):
                subprocess.check_call(["docker-compose", "up", "-d", "neo4j"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                # Fallback: start Neo4j directly
                subprocess.check_call([
                    "docker", "run", "-d",
                    "--name", "graphrag_neo4j",
                    "--restart", "unless-stopped",
                    "-p", "7474:7474",
                    "-p", "7687:7687",
                    "-e", "NEO4J_AUTH=neo4j/password",
                    "-e", "NEO4J_PLUGINS=[\"apoc\",\"graph-data-science\"]",
                    "-e", "NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*",
                    "-e", "NEO4J_dbms_security_procedures_allowlist=apoc.*,gds.*",
                    "neo4j:5.15-community"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            
            self.neo4j_container = "graphrag_neo4j"
            
            # Wait for Neo4j to be ready
            print("⏳ Waiting for Neo4j to start...")
            for i in range(30):  # Wait up to 30 seconds
                try:
                    result = subprocess.run([
                        "docker", "exec", "graphrag_neo4j", 
                        "cypher-shell", "-u", "neo4j", "-p", "password", 
                        "RETURN 1"
                    ], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        print("✅ Neo4j database ready")
                        print("🌐 Neo4j Browser: http://localhost:7474 (neo4j/password)")
                        return True
                except:
                    pass
                time.sleep(1)
            
            print("⚠️  Neo4j started but may not be fully ready yet")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to start Neo4j: {e}")
            print("GraphRAG will work in vector-only mode")
            return False
    
    def check_ollama(self, skip_check=False):
        """Check Ollama status and pull required models"""
        if skip_check:
            print("\n6️⃣  Skipping Ollama check...")
            return
        
        print("\n6️⃣  Checking Ollama and models...")
        
        # Check if Ollama is running
        try:
            result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"],
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                print("✅ Ollama server is running")
                
                # Parse available models
                try:
                    models_data = json.loads(result.stdout.decode())
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    print(f"📦 Available models: {len(available_models)}")
                    
                    # Check for required models
                    required_models = ['qwen2.5:latest', 'bge-m3:latest']
                    missing_models = []
                    
                    for model in required_models:
                        if not any(model in available for available in available_models):
                            missing_models.append(model)
                    
                    if missing_models:
                        print(f"⏳ Pulling missing models: {missing_models}")
                        for model in missing_models:
                            try:
                                print(f"   Pulling {model}...")
                                subprocess.run(["ollama", "pull", model], 
                                             check=True, timeout=300)
                                print(f"   ✅ {model} ready")
                            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                                print(f"   ⚠️  Failed to pull {model}")
                    else:
                        print("✅ All required models available")
                        
                except json.JSONDecodeError:
                    print("⚠️  Could not parse Ollama models list")
                    
            else:
                print("⚠️  Ollama server not responding")
                print("   Download from: https://ollama.ai")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  Ollama not found or not running")
            print("   Download from: https://ollama.ai")
    
    def start_fastapi_server(self, host="0.0.0.0", port=8000, reload=False):
        """Start the FastAPI server"""
        print(f"\n7️⃣  Starting FastAPI server on {host}:{port}...")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.app:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        
        try:
            # Start server process
            process = subprocess.Popen(cmd)
            self.processes.append(process)
            
            # Wait a moment for server to start
            time.sleep(2)
            
            print("✅ FastAPI server started")
            print()
            print("=" * 80)
            print("🎉 SYSTEM READY!")
            print("=" * 80)
            print(f"🌐 Main Interface:     http://localhost:{port}")
            print(f"🔧 Troubleshooting:   http://localhost:{port}/troubleshooting")
            print(f"⚔️  LLM Shootout:      http://localhost:{port}/llm-shootout")
            print(f"📊 API Documentation: http://localhost:{port}/docs")
            if self.neo4j_container:
                print(f"🗄️  Neo4j Browser:     http://localhost:7474 (neo4j/password)")
            print("=" * 80)
            print("Press Ctrl+C to stop all services")
            print()
            
            # Wait for server to finish
            process.wait()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            sys.exit(1)
    
    def run_health_check(self):
        """Run a quick health check"""
        print("\n🏥 Running health check...")
        
        # Check if server is responding
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server healthy")
            else:
                print("⚠️  API server responding but may have issues")
        except:
            print("⚠️  Could not reach API server")
        
        # Check Neo4j if available
        if self.neo4j_container:
            try:
                result = subprocess.run([
                    "docker", "exec", "graphrag_neo4j", 
                    "cypher-shell", "-u", "neo4j", "-p", "password", 
                    "RETURN 1"
                ], capture_output=True, timeout=5)
                if result.returncode == 0:
                    print("✅ Neo4j database healthy")
                else:
                    print("⚠️  Neo4j database may have issues")
            except:
                print("⚠️  Could not check Neo4j health")

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(
        description="Complete startup script for CrewAI Workflow Manager with GraphRAG"
    )
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--skip-ollama", action="store_true", 
                       help="Skip Ollama check and model pulling")
    parser.add_argument("--skip-neo4j", action="store_true", 
                       help="Skip Neo4j startup")
    parser.add_argument("--health-check", action="store_true", 
                       help="Run health check after startup")
    
    args = parser.parse_args()
    
    # Create system manager
    manager = SystemManager()
    
    try:
        # Print header
        manager.print_header()
        
        # Run startup sequence
        manager.check_python_version()
        
        docker_available = manager.check_docker() and not args.skip_neo4j
        
        manager.install_dependencies(args.skip_deps)
        
        manager.create_directories()
        
        neo4j_started = manager.start_neo4j(docker_available)
        
        manager.check_ollama(args.skip_ollama)
        
        # Run health check if requested
        if args.health_check:
            manager.run_health_check()
        
        # Start the main server (this will block until shutdown)
        manager.start_fastapi_server(args.host, args.port, args.reload)
        
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()
