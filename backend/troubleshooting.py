import asyncio
import aiohttp
import json
import sys
import os
import traceback
import time
import requests
from typing import Dict, Any, List
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.dirname(__file__))

from backend.models import WorkflowConfig, TestResult, DocumentInfo
from backend.llm_manager import LLMManager

class TroubleshootingManager:
    """Manager for running diagnostic tests"""
    
    def __init__(self):
        self.llm_manager = LLMManager()
    
    async def run_api_health_test(self, websocket_manager=None) -> Dict[str, Any]:
        """Test API health and connectivity"""
        results = {
            "test_name": "API Health Check",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_status": "running"
        }
        
        def log_message(message: str, level: str = "info"):
            if websocket_manager:
                asyncio.create_task(websocket_manager.broadcast({
                    "type": "troubleshoot_log",
                    "test": "api_health",
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }))
        
        try:
            log_message("Starting API health check tests...")
            
            # Test 1: Health endpoint
            log_message("Testing health endpoint...")
            try:
                response = requests.get("http://localhost:8000/health", timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    results["tests"].append({
                        "name": "Health Endpoint",
                        "status": "passed",
                        "message": f"Health check successful: {health_data}",
                        "response_time": response.elapsed.total_seconds() * 1000
                    })
                    log_message("✓ Health endpoint test passed", "success")
                else:
                    results["tests"].append({
                        "name": "Health Endpoint",
                        "status": "failed",
                        "message": f"Health check failed with status {response.status_code}",
                        "error": response.text
                    })
                    log_message(f"✗ Health endpoint test failed: {response.status_code}", "error")
            except Exception as e:
                results["tests"].append({
                    "name": "Health Endpoint",
                    "status": "failed",
                    "message": "Health endpoint unreachable",
                    "error": str(e)
                })
                log_message(f"✗ Health endpoint test failed: {str(e)}", "error")
            
            # Test 2: Ollama models with localhost
            log_message("Testing Ollama models with localhost...")
            try:
                response = requests.get("http://localhost:8000/ollama-models?ollama_url=http://localhost:11434", timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:
                        results["tests"].append({
                            "name": "Localhost Ollama",
                            "status": "warning",
                            "message": f"Ollama localhost connection failed: {data['error']}",
                            "models_count": 0
                        })
                        log_message(f"⚠ Localhost Ollama test warning: {data['error']}", "warning")
                    else:
                        results["tests"].append({
                            "name": "Localhost Ollama",
                            "status": "passed",
                            "message": f"Found {len(data['models'])} models on localhost",
                            "models_count": len(data['models']),
                            "models": data['models'][:5]  # First 5 models
                        })
                        log_message(f"✓ Localhost Ollama test passed: {len(data['models'])} models", "success")
                else:
                    results["tests"].append({
                        "name": "Localhost Ollama",
                        "status": "failed",
                        "message": f"Failed to get localhost Ollama models: {response.status_code}",
                        "error": response.text
                    })
                    log_message(f"✗ Localhost Ollama test failed: {response.status_code}", "error")
            except Exception as e:
                results["tests"].append({
                    "name": "Localhost Ollama",
                    "status": "failed",
                    "message": "Failed to connect to localhost Ollama",
                    "error": str(e)
                })
                log_message(f"✗ Localhost Ollama test failed: {str(e)}", "error")
            
            # Test 3: Ollama models with Docker URL
            log_message("Testing Ollama models with Docker URL...")
            try:
                response = requests.get("http://localhost:8000/ollama-models?ollama_url=http://host.docker.internal:11434", timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:
                        results["tests"].append({
                            "name": "Docker Ollama",
                            "status": "warning",
                            "message": f"Ollama Docker connection failed: {data['error']}",
                            "models_count": 0
                        })
                        log_message(f"⚠ Docker Ollama test warning: {data['error']}", "warning")
                    else:
                        results["tests"].append({
                            "name": "Docker Ollama",
                            "status": "passed",
                            "message": f"Found {len(data['models'])} models on Docker",
                            "models_count": len(data['models']),
                            "models": data['models'][:5]  # First 5 models
                        })
                        log_message(f"✓ Docker Ollama test passed: {len(data['models'])} models", "success")
                else:
                    results["tests"].append({
                        "name": "Docker Ollama",
                        "status": "failed",
                        "message": f"Failed to get Docker Ollama models: {response.status_code}",
                        "error": response.text
                    })
                    log_message(f"✗ Docker Ollama test failed: {response.status_code}", "error")
            except Exception as e:
                results["tests"].append({
                    "name": "Docker Ollama",
                    "status": "failed",
                    "message": "Failed to connect to Docker Ollama",
                    "error": str(e)
                })
                log_message(f"✗ Docker Ollama test failed: {str(e)}", "error")
            
            # Determine overall status
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "passed")
            failed_tests = sum(1 for test in results["tests"] if test["status"] == "failed")
            warning_tests = sum(1 for test in results["tests"] if test["status"] == "warning")
            
            if failed_tests == 0 and warning_tests == 0:
                results["overall_status"] = "passed"
                log_message("✓ All API health tests passed!", "success")
            elif failed_tests == 0:
                results["overall_status"] = "warning"
                log_message("⚠ API health tests completed with warnings", "warning")
            else:
                results["overall_status"] = "failed"
                log_message("✗ Some API health tests failed", "error")
            
            results["summary"] = {
                "total_tests": len(results["tests"]),
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            }
            
        except Exception as e:
            log_message(f"✗ API health test suite failed: {str(e)}", "error")
            results["overall_status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    async def run_docker_ollama_test(self, websocket_manager=None) -> Dict[str, Any]:
        """Test Docker Ollama connection specifically"""
        results = {
            "test_name": "Docker Ollama Connection Test",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_status": "running"
        }
        
        def log_message(message: str, level: str = "info"):
            if websocket_manager:
                asyncio.create_task(websocket_manager.broadcast({
                    "type": "troubleshoot_log",
                    "test": "docker_ollama",
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }))
        
        try:
            log_message("Starting Docker Ollama connection test...")
            
            ollama_url = "http://host.docker.internal:11434"
            model_name = "bge-m3:latest"
            
            log_message(f"Testing Ollama URL: {ollama_url}")
            log_message(f"Test model: {model_name}")
            
            async with aiohttp.ClientSession() as session:
                # Test connection to Ollama
                log_message("Testing connection to Ollama server...")
                try:
                    async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            models_data = await response.json()
                            available_models = [model["name"] for model in models_data.get("models", [])]
                            
                            results["tests"].append({
                                "name": "Docker Ollama Connection",
                                "status": "passed",
                                "message": f"Successfully connected to Docker Ollama",
                                "models_found": len(available_models),
                                "available_models": available_models[:10]  # First 10 models
                            })
                            log_message(f"✓ Docker Ollama connection successful: {len(available_models)} models found", "success")
                            
                            # Test specific model if available
                            if model_name in available_models:
                                log_message(f"Testing model {model_name}...")
                                test_payload = {
                                    "model": model_name,
                                    "prompt": "Hello, this is a test."
                                }
                                
                                try:
                                    async with session.post(
                                        f"{ollama_url}/api/embeddings",
                                        json=test_payload,
                                        timeout=aiohttp.ClientTimeout(total=300)
                                    ) as model_response:
                                        if model_response.status == 200:
                                            result = await model_response.json()
                                            if "embedding" in result and result["embedding"]:
                                                results["tests"].append({
                                                    "name": f"Model Test ({model_name})",
                                                    "status": "passed",
                                                    "message": f"Model {model_name} working correctly",
                                                    "embedding_length": len(result['embedding'])
                                                })
                                                log_message(f"✓ Model {model_name} test passed", "success")
                                            else:
                                                results["tests"].append({
                                                    "name": f"Model Test ({model_name})",
                                                    "status": "failed",
                                                    "message": f"Model {model_name} returned invalid response",
                                                    "error": "No embedding data"
                                                })
                                                log_message(f"✗ Model {model_name} test failed: no embedding data", "error")
                                        else:
                                            error_text = await model_response.text()
                                            results["tests"].append({
                                                "name": f"Model Test ({model_name})",
                                                "status": "failed",
                                                "message": f"Model {model_name} test failed",
                                                "error": error_text
                                            })
                                            log_message(f"✗ Model {model_name} test failed: {error_text}", "error")
                                except asyncio.TimeoutError:
                                    results["tests"].append({
                                        "name": f"Model Test ({model_name})",
                                        "status": "failed",
                                        "message": f"Model {model_name} test timeout",
                                        "error": "Timeout after 300 seconds"
                                    })
                                    log_message(f"✗ Model {model_name} test timeout", "error")
                            else:
                                results["tests"].append({
                                    "name": f"Model Availability ({model_name})",
                                    "status": "warning",
                                    "message": f"Model {model_name} not found",
                                    "available_models": available_models[:5]
                                })
                                log_message(f"⚠ Model {model_name} not found in available models", "warning")
                        else:
                            results["tests"].append({
                                "name": "Docker Ollama Connection",
                                "status": "failed",
                                "message": f"Failed to connect: HTTP {response.status}",
                                "error": await response.text()
                            })
                            log_message(f"✗ Docker Ollama connection failed: HTTP {response.status}", "error")
                except asyncio.TimeoutError:
                    results["tests"].append({
                        "name": "Docker Ollama Connection",
                        "status": "failed",
                        "message": "Connection timeout - Ollama server may not be accessible from Docker",
                        "error": "Timeout after 60 seconds"
                    })
                    log_message("✗ Docker Ollama connection timeout", "error")
                except Exception as e:
                    results["tests"].append({
                        "name": "Docker Ollama Connection",
                        "status": "failed",
                        "message": "Connection error",
                        "error": str(e)
                    })
                    log_message(f"✗ Docker Ollama connection error: {str(e)}", "error")
            
            # Determine overall status
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "passed")
            failed_tests = sum(1 for test in results["tests"] if test["status"] == "failed")
            warning_tests = sum(1 for test in results["tests"] if test["status"] == "warning")
            
            if failed_tests == 0 and warning_tests == 0:
                results["overall_status"] = "passed"
                log_message("✓ All Docker Ollama tests passed!", "success")
            elif failed_tests == 0:
                results["overall_status"] = "warning"
                log_message("⚠ Docker Ollama tests completed with warnings", "warning")
            else:
                results["overall_status"] = "failed"
                log_message("✗ Some Docker Ollama tests failed", "error")
            
            results["summary"] = {
                "total_tests": len(results["tests"]),
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            }
            
        except Exception as e:
            log_message(f"✗ Docker Ollama test suite failed: {str(e)}", "error")
            results["overall_status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    async def run_model_debug_test(self, model_name: str = "bge-m3:latest", websocket_manager=None) -> Dict[str, Any]:
        """Run detailed model debugging"""
        results = {
            "test_name": f"Model Debug Test ({model_name})",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_status": "running"
        }
        
        def log_message(message: str, level: str = "info"):
            if websocket_manager:
                asyncio.create_task(websocket_manager.broadcast({
                    "type": "troubleshoot_log",
                    "test": "model_debug",
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }))
        
        try:
            log_message(f"Starting model debug test for: {model_name}")
            
            ollama_url = "http://localhost:11434"
            
            async with aiohttp.ClientSession() as session:
                # Check available models
                log_message("Checking available models...")
                async with session.get(f"{ollama_url}/api/tags") as response:
                    if response.status != 200:
                        results["tests"].append({
                            "name": "Ollama Server Connection",
                            "status": "failed",
                            "message": f"Cannot connect to Ollama server - HTTP {response.status}",
                            "error": await response.text()
                        })
                        log_message(f"✗ Cannot connect to Ollama server: HTTP {response.status}", "error")
                        results["overall_status"] = "failed"
                        return results
                    
                    models_data = await response.json()
                    available_models = [model["name"] for model in models_data.get("models", [])]
                    
                    results["tests"].append({
                        "name": "Available Models Check",
                        "status": "passed",
                        "message": f"Found {len(available_models)} available models",
                        "available_models": available_models
                    })
                    log_message(f"✓ Found {len(available_models)} available models", "success")
                    
                    # Check if target model exists
                    model_base_name = model_name.split(":")[0]
                    available_model_names = [model.split(":")[0] for model in available_models]
                    
                    if model_base_name not in available_model_names:
                        results["tests"].append({
                            "name": "Target Model Availability",
                            "status": "failed",
                            "message": f"Model {model_name} not found",
                            "available_alternatives": available_model_names[:5]
                        })
                        log_message(f"✗ Model {model_name} not found", "error")
                        results["overall_status"] = "failed"
                        return results
                    
                    # Find actual model name
                    actual_model_name = model_name if model_name in available_models else model_base_name
                    
                    results["tests"].append({
                        "name": "Target Model Availability",
                        "status": "passed",
                        "message": f"Model found: {actual_model_name}",
                        "actual_model_name": actual_model_name
                    })
                    log_message(f"✓ Model found: {actual_model_name}", "success")
                    
                    # Get model details
                    log_message("Getting model details...")
                    model_details = None
                    for model in models_data.get("models", []):
                        if model["name"] == actual_model_name:
                            model_details = model.get("details", {})
                            break
                    
                    if model_details:
                        family = model_details.get("family", "").lower()
                        families = model_details.get("families", [])
                        is_embedding_model = (family == "bert" or "bert" in families or 
                                            "bge" in model_base_name.lower() or 
                                            "embed" in model_base_name.lower())
                        
                        results["tests"].append({
                            "name": "Model Details Analysis",
                            "status": "passed",
                            "message": f"Model type: {'Embedding' if is_embedding_model else 'Text Generation'}",
                            "model_details": model_details,
                            "is_embedding_model": is_embedding_model
                        })
                        log_message(f"✓ Model analysis complete: {'Embedding' if is_embedding_model else 'Text Generation'} model", "success")
                        
                        # Test the model
                        log_message(f"Testing model functionality...")
                        if is_embedding_model:
                            test_payload = {
                                "model": actual_model_name,
                                "prompt": "Hello, this is a test."
                            }
                            
                            async with session.post(
                                f"{ollama_url}/api/embeddings",
                                json=test_payload
                            ) as test_response:
                                if test_response.status == 200:
                                    result = await test_response.json()
                                    if "embedding" in result and result["embedding"]:
                                        results["tests"].append({
                                            "name": "Model Functionality Test",
                                            "status": "passed",
                                            "message": "Embedding model working correctly",
                                            "embedding_length": len(result['embedding']),
                                            "sample_embedding": result['embedding'][:5]
                                        })
                                        log_message("✓ Model functionality test passed", "success")
                                    else:
                                        results["tests"].append({
                                            "name": "Model Functionality Test",
                                            "status": "failed",
                                            "message": "No embedding data in response",
                                            "response": result
                                        })
                                        log_message("✗ Model functionality test failed: no embedding data", "error")
                                else:
                                    error_text = await test_response.text()
                                    results["tests"].append({
                                        "name": "Model Functionality Test",
                                        "status": "failed",
                                        "message": "Model test failed",
                                        "error": error_text
                                    })
                                    log_message(f"✗ Model functionality test failed: {error_text}", "error")
                        else:
                            # Text generation model test
                            test_payload = {
                                "model": actual_model_name,
                                "prompt": "Hello, this is a test.",
                                "stream": False,
                                "options": {"num_predict": 10}
                            }
                            
                            async with session.post(
                                f"{ollama_url}/api/generate",
                                json=test_payload
                            ) as test_response:
                                if test_response.status == 200:
                                    result = await test_response.json()
                                    results["tests"].append({
                                        "name": "Model Functionality Test",
                                        "status": "passed",
                                        "message": "Text generation model working correctly",
                                        "response": result.get('response', 'No response')
                                    })
                                    log_message("✓ Model functionality test passed", "success")
                                else:
                                    error_text = await test_response.text()
                                    results["tests"].append({
                                        "name": "Model Functionality Test",
                                        "status": "failed",
                                        "message": "Model test failed",
                                        "error": error_text
                                    })
                                    log_message(f"✗ Model functionality test failed: {error_text}", "error")
                    else:
                        results["tests"].append({
                            "name": "Model Details Analysis",
                            "status": "warning",
                            "message": "Could not retrieve model details",
                        })
                        log_message("⚠ Could not retrieve model details", "warning")
            
            # Determine overall status
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "passed")
            failed_tests = sum(1 for test in results["tests"] if test["status"] == "failed")
            warning_tests = sum(1 for test in results["tests"] if test["status"] == "warning")
            
            if failed_tests == 0 and warning_tests == 0:
                results["overall_status"] = "passed"
                log_message("✓ All model debug tests passed!", "success")
            elif failed_tests == 0:
                results["overall_status"] = "warning"
                log_message("⚠ Model debug tests completed with warnings", "warning")
            else:
                results["overall_status"] = "failed"
                log_message("✗ Some model debug tests failed", "error")
            
            results["summary"] = {
                "total_tests": len(results["tests"]),
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            }
            
        except Exception as e:
            log_message(f"✗ Model debug test suite failed: {str(e)}", "error")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        return results
    
    async def run_workflow_model_test(self, config: WorkflowConfig, websocket_manager=None) -> Dict[str, Any]:
        """Test workflow model setup"""
        results = {
            "test_name": "Workflow Model Test",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_status": "running"
        }
        
        def log_message(message: str, level: str = "info"):
            if websocket_manager:
                asyncio.create_task(websocket_manager.broadcast({
                    "type": "troubleshoot_log",
                    "test": "workflow_model",
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }))
        
        try:
            log_message("Starting workflow model test...")
            log_message(f"Data generation model: {config.data_generation_model}")
            log_message(f"Embedding model: {config.embedding_model}")
            log_message(f"Ollama URL: {config.ollama_url}")
            
            # Test models using LLM manager
            log_message("Testing models with LLM manager...")
            test_results = await self.llm_manager.test_models(config)
            
            for model_type, result in test_results.items():
                status = "passed" if result.success else "failed"
                test_entry = {
                    "name": f"{model_type.title()} Model Test",
                    "status": status,
                    "message": result.message
                }
                
                if result.error:
                    test_entry["error"] = result.error
                if result.response_time:
                    test_entry["response_time"] = result.response_time
                
                results["tests"].append(test_entry)
                
                if result.success:
                    log_message(f"✓ {model_type} model test passed", "success")
                else:
                    log_message(f"✗ {model_type} model test failed: {result.message}", "error")
            
            # Determine overall status
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "passed")
            failed_tests = sum(1 for test in results["tests"] if test["status"] == "failed")
            warning_tests = sum(1 for test in results["tests"] if test["status"] == "warning")
            
            if failed_tests == 0 and warning_tests == 0:
                results["overall_status"] = "passed"
                log_message("✓ All workflow model tests passed!", "success")
            elif failed_tests == 0:
                results["overall_status"] = "warning"
                log_message("⚠ Workflow model tests completed with warnings", "warning")
            else:
                results["overall_status"] = "failed"
                log_message("✗ Some workflow model tests failed", "error")
            
            results["summary"] = {
                "total_tests": len(results["tests"]),
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            }
            
        except Exception as e:
            log_message(f"✗ Workflow model test suite failed: {str(e)}", "error")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        return results
    
    async def run_llm_manager_debug(self, config: WorkflowConfig, websocket_manager=None) -> Dict[str, Any]:
        """Run comprehensive LLM manager debugging"""
        results = {
            "test_name": "LLM Manager Debug",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_status": "running"
        }
        
        def log_message(message: str, level: str = "info"):
            if websocket_manager:
                asyncio.create_task(websocket_manager.broadcast({
                    "type": "troubleshoot_log",
                    "test": "llm_debug",
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }))
        
        try:
            log_message("Starting LLM manager debug test...")
            
            # Test Ollama connection
            log_message("Testing Ollama server connection...")
            try:
                models = await self.llm_manager.get_ollama_models(config.ollama_url)
                results["tests"].append({
                    "name": "Ollama Server Connection",
                    "status": "passed",
                    "message": f"Successfully connected to Ollama server",
                    "models_count": len(models),
                    "ollama_url": config.ollama_url
                })
                log_message(f"✓ Ollama server connection successful: {len(models)} models", "success")
            except Exception as e:
                results["tests"].append({
                    "name": "Ollama Server Connection",
                    "status": "failed",
                    "message": "Failed to connect to Ollama server",
                    "error": str(e),
                    "ollama_url": config.ollama_url
                })
                log_message(f"✗ Ollama server connection failed: {str(e)}", "error")
            
            # Test individual models
            if config.embedding_model:
                log_message(f"Testing embedding model: {config.embedding_model}")
                try:
                    embedding_result = await self.llm_manager._test_model(config.embedding_model)
                    status = "passed" if embedding_result.success else "failed"
                    results["tests"].append({
                        "name": "Embedding Model Test",
                        "status": status,
                        "message": embedding_result.message,
                        "model": config.embedding_model,
                        "error": embedding_result.error if embedding_result.error else None,
                        "response_time": embedding_result.response_time
                    })
                    
                    if embedding_result.success:
                        log_message("✓ Embedding model test passed", "success")
                    else:
                        log_message(f"✗ Embedding model test failed: {embedding_result.message}", "error")
                except Exception as e:
                    results["tests"].append({
                        "name": "Embedding Model Test",
                        "status": "failed",
                        "message": "Exception during embedding model test",
                        "error": str(e)
                    })
                    log_message(f"✗ Embedding model test exception: {str(e)}", "error")
            
            if config.data_generation_model:
                log_message(f"Testing data generation model: {config.data_generation_model}")
                try:
                    datagen_result = await self.llm_manager._test_model(config.data_generation_model)
                    status = "passed" if datagen_result.success else "failed"
                    results["tests"].append({
                        "name": "Data Generation Model Test",
                        "status": status,
                        "message": datagen_result.message,
                        "model": config.data_generation_model,
                        "error": datagen_result.error if datagen_result.error else None,
                        "response_time": datagen_result.response_time
                    })
                    
                    if datagen_result.success:
                        log_message("✓ Data generation model test passed", "success")
                    else:
                        log_message(f"✗ Data generation model test failed: {datagen_result.message}", "error")
                except Exception as e:
                    results["tests"].append({
                        "name": "Data Generation Model Test",
                        "status": "failed",
                        "message": "Exception during data generation model test",
                        "error": str(e)
                    })
                    log_message(f"✗ Data generation model test exception: {str(e)}", "error")
            
            # Determine overall status
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "passed")
            failed_tests = sum(1 for test in results["tests"] if test["status"] == "failed")
            warning_tests = sum(1 for test in results["tests"] if test["status"] == "warning")
            
            if failed_tests == 0 and warning_tests == 0:
                results["overall_status"] = "passed"
                log_message("✓ All LLM manager debug tests passed!", "success")
            elif failed_tests == 0:
                results["overall_status"] = "warning"
                log_message("⚠ LLM manager debug tests completed with warnings", "warning")
            else:
                results["overall_status"] = "failed"
                log_message("✗ Some LLM manager debug tests failed", "error")
            
            results["summary"] = {
                "total_tests": len(results["tests"]),
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            }
            
        except Exception as e:
            log_message(f"✗ LLM manager debug test suite failed: {str(e)}", "error")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        return results
    
    async def run_crew_workflow_test(self, config: WorkflowConfig, websocket_manager=None) -> Dict[str, Any]:
        """Test CrewAI workflow execution with proper model configuration"""
        results = {
            "test_name": "CrewAI Workflow Test",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_status": "running"
        }
        
        def log_message(message: str, level: str = "info"):
            if websocket_manager:
                asyncio.create_task(websocket_manager.broadcast({
                    "type": "troubleshoot_log",
                    "test": "crew_workflow",
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }))
        
        try:
            log_message("Starting CrewAI workflow test...")
            
            # Import CrewAI crew
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
                from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew
                
                results["tests"].append({
                    "name": "CrewAI Import",
                    "status": "passed",
                    "message": "Successfully imported CrewAI crew class"
                })
                log_message("✓ CrewAI crew class imported successfully", "success")
            except Exception as e:
                results["tests"].append({
                    "name": "CrewAI Import",
                    "status": "failed",
                    "message": "Failed to import CrewAI crew class",
                    "error": str(e)
                })
                log_message(f"✗ CrewAI import failed: {str(e)}", "error")
                results["overall_status"] = "failed"
                return results
            
            # Create crew configuration
            crew_config = {
                "data_generation_model": config.data_generation_model,
                "embedding_model": config.embedding_model,
                "reranking_model": config.reranking_model,
                "openai_api_key": config.openai_api_key,
                "ollama_url": config.ollama_url,
                "enable_gpu_optimization": config.enable_gpu_optimization
            }
            
            log_message(f"Configuration: {crew_config}")
            
            # Test crew creation
            try:
                crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=crew_config)
                
                results["tests"].append({
                    "name": "CrewAI Crew Creation",
                    "status": "passed",
                    "message": "CrewAI crew instance created successfully"
                })
                log_message("✓ CrewAI crew instance created successfully", "success")
                
                # Check LLM initialization
                if crew_instance.data_generation_llm is not None:
                    results["tests"].append({
                        "name": "Data Generation LLM",
                        "status": "passed",
                        "message": f"Data generation LLM initialized: {crew_instance.data_generation_llm.model}",
                        "model": crew_instance.data_generation_llm.model
                    })
                    log_message(f"✓ Data generation LLM: {crew_instance.data_generation_llm.model}", "success")
                else:
                    results["tests"].append({
                        "name": "Data Generation LLM",
                        "status": "failed",
                        "message": "Data generation LLM failed to initialize"
                    })
                    log_message("✗ Data generation LLM failed to initialize", "error")
                
                if crew_instance.embedding_llm is not None:
                    results["tests"].append({
                        "name": "Embedding LLM",
                        "status": "passed",
                        "message": f"Embedding LLM initialized: {crew_instance.embedding_llm.model}",
                        "model": crew_instance.embedding_llm.model
                    })
                    log_message(f"✓ Embedding LLM: {crew_instance.embedding_llm.model}", "success")
                else:
                    results["tests"].append({
                        "name": "Embedding LLM",
                        "status": "failed",
                        "message": "Embedding LLM failed to initialize"
                    })
                    log_message("✗ Embedding LLM failed to initialize", "error")
                
                if crew_instance.reranking_llm is not None:
                    results["tests"].append({
                        "name": "Reranking LLM",
                        "status": "passed",
                        "message": f"Reranking LLM initialized: {crew_instance.reranking_llm.model}",
                        "model": crew_instance.reranking_llm.model
                    })
                    log_message(f"✓ Reranking LLM: {crew_instance.reranking_llm.model}", "success")
                else:
                    results["tests"].append({
                        "name": "Reranking LLM",
                        "status": "warning",
                        "message": "Reranking LLM not initialized (optional)"
                    })
                    log_message("⚠ Reranking LLM not initialized (optional)", "warning")
                
            except Exception as e:
                results["tests"].append({
                    "name": "CrewAI Crew Creation",
                    "status": "failed",
                    "message": "Failed to create CrewAI crew instance",
                    "error": str(e)
                })
                log_message(f"✗ CrewAI crew creation failed: {str(e)}", "error")
                results["overall_status"] = "failed"
                return results
            
            # Test quick workflow execution (minimal)
            try:
                log_message("Testing quick workflow execution...")
                
                inputs = {
                    "documents": [],  # Empty for quick test
                    "workflow_type": "test",
                    "data_generation_model": config.data_generation_model,
                    "embedding_model": config.embedding_model,
                    "reranking_model": config.reranking_model,
                    "enable_gpu_optimization": False  # Disable for quick test
                }
                
                # Run in thread to avoid blocking
                import concurrent.futures
                loop = asyncio.get_event_loop()
                
                def run_crew():
                    try:
                        result = crew_instance.crew().kickoff(inputs=inputs)
                        return {"success": True, "result": str(result)[:200]}
                    except Exception as e:
                        return {"success": False, "error": str(e)}
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    crew_result = await loop.run_in_executor(executor, run_crew)
                
                if crew_result["success"]:
                    results["tests"].append({
                        "name": "Quick Workflow Execution",
                        "status": "passed",
                        "message": "CrewAI workflow executed successfully",
                        "result_preview": crew_result["result"]
                    })
                    log_message("✓ CrewAI workflow executed successfully", "success")
                else:
                    results["tests"].append({
                        "name": "Quick Workflow Execution",
                        "status": "failed",
                        "message": "CrewAI workflow execution failed",
                        "error": crew_result["error"]
                    })
                    log_message(f"✗ CrewAI workflow execution failed: {crew_result['error']}", "error")
                
            except Exception as e:
                results["tests"].append({
                    "name": "Quick Workflow Execution",
                    "status": "failed",
                    "message": "Exception during workflow execution",
                    "error": str(e)
                })
                log_message(f"✗ Workflow execution exception: {str(e)}", "error")
            
            # Determine overall status
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "passed")
            failed_tests = sum(1 for test in results["tests"] if test["status"] == "failed")
            warning_tests = sum(1 for test in results["tests"] if test["status"] == "warning")
            
            if failed_tests == 0 and warning_tests == 0:
                results["overall_status"] = "passed"
                log_message("✓ All CrewAI workflow tests passed!", "success")
            elif failed_tests == 0:
                results["overall_status"] = "warning"
                log_message("⚠ CrewAI workflow tests completed with warnings", "warning")
            else:
                results["overall_status"] = "failed"
                log_message("✗ Some CrewAI workflow tests failed", "error")
            
            results["summary"] = {
                "total_tests": len(results["tests"]),
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            }
            
        except Exception as e:
            log_message(f"✗ CrewAI workflow test suite failed: {str(e)}", "error")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        return results
    
    async def run_ollama_workflow_test(self, config: WorkflowConfig, websocket_manager=None) -> Dict[str, Any]:
        """Test Ollama workflow configuration with dynamic model selection"""
        results = {
            "test_name": "Ollama Workflow Test",
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "overall_status": "running"
        }
        
        def log_message(message: str, level: str = "info"):
            if websocket_manager:
                asyncio.create_task(websocket_manager.broadcast({
                    "type": "troubleshoot_log",
                    "test": "ollama_workflow",
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().isoformat()
                }))
        
        try:
            log_message("Starting Ollama workflow configuration test...")
            
            # Test LLM Manager with configuration
            log_message("Testing LLM manager with Ollama configuration...")
            try:
                test_results = await self.llm_manager.test_models(config)
                
                for model_type, result in test_results.items():
                    status = "passed" if result.success else "failed"
                    test_entry = {
                        "name": f"{model_type.title()} Model Test",
                        "status": status,
                        "message": result.message,
                        "model": getattr(config, f"{model_type}_model", "Unknown")
                    }
                    
                    if result.error:
                        test_entry["error"] = result.error
                    if result.response_time:
                        test_entry["response_time"] = result.response_time
                    
                    results["tests"].append(test_entry)
                    
                    if result.success:
                        log_message(f"✓ {model_type} model test passed", "success")
                    else:
                        log_message(f"✗ {model_type} model test failed: {result.message}", "error")
                
            except Exception as e:
                results["tests"].append({
                    "name": "LLM Manager Test",
                    "status": "failed",
                    "message": "Exception during LLM manager testing",
                    "error": str(e)
                })
                log_message(f"✗ LLM manager test exception: {str(e)}", "error")
            
            # Test CrewAI crew initialization without OpenAI dependency
            try:
                log_message("Testing CrewAI crew initialization...")
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
                from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew
                
                crew_config = {
                    "data_generation_model": config.data_generation_model,
                    "embedding_model": config.embedding_model,
                    "reranking_model": config.reranking_model,
                    "openai_api_key": config.openai_api_key,
                    "ollama_url": config.ollama_url,
                    "enable_gpu_optimization": config.enable_gpu_optimization
                }
                
                crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=crew_config)
                
                results["tests"].append({
                    "name": "CrewAI Initialization",
                    "status": "passed",
                    "message": "CrewAI crew created without requiring OpenAI API key",
                    "data_gen_model": crew_instance.data_generation_llm.model if crew_instance.data_generation_llm else None,
                    "embedding_model": crew_instance.embedding_llm.model if crew_instance.embedding_llm else None
                })
                log_message("✓ CrewAI crew created without requiring OpenAI API key", "success")
                
            except Exception as e:
                results["tests"].append({
                    "name": "CrewAI Initialization",
                    "status": "failed",
                    "message": "Failed to create CrewAI crew",
                    "error": str(e)
                })
                log_message(f"✗ CrewAI initialization failed: {str(e)}", "error")
            
            # Test OpenAI model rejection when no API key provided
            if not config.openai_api_key:
                try:
                    log_message("Testing OpenAI model rejection without API key...")
                    
                    # This should fail gracefully
                    openai_config = WorkflowConfig(
                        data_generation_model="openai:gpt-3.5-turbo",
                        embedding_model=config.embedding_model,
                        reranking_model=config.reranking_model,
                        openai_api_key=None,
                        ollama_url=config.ollama_url,
                        enable_gpu_optimization=config.enable_gpu_optimization
                    )
                    
                    openai_test = await self.llm_manager.test_models(openai_config)
                    
                    if not openai_test["data_generation"].success:
                        results["tests"].append({
                            "name": "OpenAI Model Rejection",
                            "status": "passed",
                            "message": "Correctly rejected OpenAI model without API key",
                            "error_message": openai_test["data_generation"].error
                        })
                        log_message("✓ Correctly rejected OpenAI model without API key", "success")
                    else:
                        results["tests"].append({
                            "name": "OpenAI Model Rejection",
                            "status": "warning",
                            "message": "OpenAI model test unexpectedly passed without API key"
                        })
                        log_message("⚠ OpenAI model test unexpectedly passed without API key", "warning")
                        
                except Exception as e:
                    results["tests"].append({
                        "name": "OpenAI Model Rejection",
                        "status": "passed",
                        "message": "OpenAI model correctly failed without API key",
                        "error": str(e)
                    })
                    log_message("✓ OpenAI model correctly failed without API key", "success")
            
            # Determine overall status
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "passed")
            failed_tests = sum(1 for test in results["tests"] if test["status"] == "failed")
            warning_tests = sum(1 for test in results["tests"] if test["status"] == "warning")
            
            if failed_tests == 0 and warning_tests == 0:
                results["overall_status"] = "passed"
                log_message("✓ All Ollama workflow tests passed!", "success")
            elif failed_tests == 0:
                results["overall_status"] = "warning"
                log_message("⚠ Ollama workflow tests completed with warnings", "warning")
            else:
                results["overall_status"] = "failed"
                log_message("✗ Some Ollama workflow tests failed", "error")
            
            results["summary"] = {
                "total_tests": len(results["tests"]),
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            }
            
        except Exception as e:
            log_message(f"✗ Ollama workflow test suite failed: {str(e)}", "error")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        return results
