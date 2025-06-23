import asyncio
import aiohttp
import time
import logging
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.models import WorkflowConfig, TestResult

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebugLLMManager:
    """Debug version of LLM manager with extensive logging"""
    
    def __init__(self):
        self.ollama_base_url = "http://host.docker.internal:11434"
    
    async def test_models(self, config: WorkflowConfig):
        """Test all configured models with debug output"""
        results = {}
        
        # Update Ollama URL if provided
        if config.ollama_url:
            self.ollama_base_url = config.ollama_url
            print(f"Updated Ollama URL to: {self.ollama_base_url}")
        
        # Test embedding model
        if config.embedding_model:
            print(f"\nTesting embedding model: {config.embedding_model}")
            results["embedding"] = await self._test_model(config.embedding_model)
        
        return results
    
    async def _test_model(self, model_spec: str) -> TestResult:
        """Test a specific model with debug output"""
        print(f"Testing model spec: {model_spec}")
        
        try:
            provider, model_name = model_spec.split(":", 1)
            print(f"Provider: {provider}, Model name: {model_name}")
            
            if provider == "ollama":
                return await self._test_ollama_model(model_name)
            else:
                return TestResult(
                    success=False,
                    message=f"Unknown provider: {provider}",
                    error=f"Provider {provider} is not supported"
                )
        except Exception as e:
            print(f"Error parsing model spec: {str(e)}")
            return TestResult(
                success=False,
                message=f"Failed to test model: {str(e)}",
                error=str(e)
            )
    
    async def _test_ollama_model(self, model_name: str) -> TestResult:
        """Test Ollama model with extensive debug output"""
        print(f"Testing Ollama model: {model_name}")
        print(f"Using Ollama URL: {self.ollama_base_url}")
        
        try:
            # Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # First check if model exists
                print("Step 1: Checking if Ollama server is accessible...")
                
                try:
                    async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                        print(f"Ollama server response status: {response.status}")
                        
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Ollama server error: {error_text}")
                            return TestResult(
                                success=False,
                                message="Cannot connect to Ollama server",
                                error=f"HTTP {response.status}: {error_text}"
                            )
                        
                        models_data = await response.json()
                        available_models = [model["name"] for model in models_data.get("models", [])]
                        print(f"Available models: {available_models}")
                        
                        available_model_names = [model.split(":")[0] for model in available_models]
                        print(f"Available model base names: {available_model_names}")
                        
                        # Check if the model name (without tag) exists
                        model_base_name = model_name.split(":")[0]
                        print(f"Looking for model base name: {model_base_name}")
                        
                        if model_base_name not in available_model_names:
                            return TestResult(
                                success=False,
                                message=f"Model {model_name} not found in Ollama. Available models: {', '.join(available_model_names)}",
                                error="Model not found"
                            )
                        
                        # Find the actual model name to use
                        actual_model_name = model_name
                        if model_name not in available_models:
                            actual_model_name = model_base_name
                            print(f"Using base model name: {actual_model_name}")
                        else:
                            print(f"Using exact model name: {actual_model_name}")
                        
                        # Get model details
                        print("Step 2: Getting model details...")
                        model_details = None
                        for model in models_data.get("models", []):
                            if model["name"] == actual_model_name:
                                model_details = model.get("details", {})
                                print(f"Model details: {model_details}")
                                break
                        
                        # Check if this is an embedding model
                        is_embedding_model = False
                        if model_details:
                            family = model_details.get("family", "").lower()
                            families = model_details.get("families", [])
                            print(f"Model family: {family}, families: {families}")
                            
                            if family == "bert" or "bert" in families or "bge" in model_base_name.lower() or "embed" in model_base_name.lower():
                                is_embedding_model = True
                        
                        print(f"Is embedding model: {is_embedding_model}")
                
                except asyncio.TimeoutError:
                    print("Timeout connecting to Ollama server")
                    return TestResult(
                        success=False,
                        message="Timeout connecting to Ollama server",
                        error="Connection timeout"
                    )
                except Exception as e:
                    print(f"Error connecting to Ollama server: {str(e)}")
                    return TestResult(
                        success=False,
                        message=f"Error connecting to Ollama server: {str(e)}",
                        error=str(e)
                    )
                
                # Test the model
                print("Step 3: Testing model...")
                if is_embedding_model:
                    print("Testing as embedding model...")
                    test_payload = {
                        "model": actual_model_name,
                        "prompt": "Hello, this is a test."
                    }
                    print(f"Test payload: {test_payload}")
                    
                    try:
                        async with session.post(
                            f"{self.ollama_base_url}/api/embeddings",
                            json=test_payload
                        ) as response:
                            print(f"Embedding test response status: {response.status}")
                            
                            if response.status == 200:
                                result = await response.json()
                                if "embedding" in result and result["embedding"]:
                                    print(f"Embedding test successful, length: {len(result['embedding'])}")
                                    return TestResult(
                                        success=True,
                                        message=f"Ollama embedding model {actual_model_name} is working correctly"
                                    )
                                else:
                                    print("No embedding data in response")
                                    return TestResult(
                                        success=False,
                                        message=f"Ollama embedding model {actual_model_name} returned invalid response",
                                        error="No embedding data in response"
                                    )
                            else:
                                error_text = await response.text()
                                print(f"Embedding test failed: {error_text}")
                                return TestResult(
                                    success=False,
                                    message=f"Ollama embedding model {actual_model_name} test failed",
                                    error=error_text
                                )
                    except asyncio.TimeoutError:
                        print("Timeout testing embedding model")
                        return TestResult(
                            success=False,
                            message="Timeout testing embedding model",
                            error="Model test timeout"
                        )
                    except Exception as e:
                        print(f"Error testing embedding model: {str(e)}")
                        return TestResult(
                            success=False,
                            message=f"Error testing embedding model: {str(e)}",
                            error=str(e)
                        )
                else:
                    print("Testing as text generation model...")
                    # Similar logic for text generation models
                    return TestResult(
                        success=False,
                        message="Text generation model testing not implemented in debug version",
                        error="Not implemented"
                    )
        
        except Exception as e:
            print(f"General error testing Ollama model: {str(e)}")
            import traceback
            traceback.print_exc()
            return TestResult(
                success=False,
                message=f"Failed to test Ollama model {model_name}: {str(e)}",
                error=str(e)
            )

async def debug_test():
    """Run debug test"""
    config = WorkflowConfig(
        data_generation_model="ollama:llama3.3:latest",
        embedding_model="ollama:bge-m3:latest",
        reranking_model=None,
        openai_api_key=None,
        ollama_url="http://host.docker.internal:11434",
        enable_gpu_optimization=True,
        documents=[],
        workflow_type="full"
    )
    
    debug_manager = DebugLLMManager()
    results = await debug_manager.test_models(config)
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    for model_type, result in results.items():
        print(f"{model_type}: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Message: {result.message}")
        if result.error:
            print(f"  Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_test())
