import asyncio
import aiohttp
import json

async def test_docker_ollama_connection():
    """Test Ollama connection from Docker environment"""
    ollama_url = "http://host.docker.internal:11434"
    model_name = "bge-m3:latest"
    
    print(f"Testing Docker Ollama connection...")
    print(f"Ollama URL: {ollama_url}")
    print(f"Model: {model_name}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test connection to Ollama
            print("\n1. Testing connection to Ollama server...")
            try:
                async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    print(f"Connection status: {response.status}")
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [model["name"] for model in models_data.get("models", [])]
                        print(f"Available models: {available_models}")
                        
                        if model_name in available_models:
                            print(f"✓ Model {model_name} found")
                        else:
                            print(f"✗ Model {model_name} not found")
                            return
                    else:
                        print(f"✗ Failed to connect: HTTP {response.status}")
                        return
            except asyncio.TimeoutError:
                print("✗ Connection timeout - Ollama server may not be accessible from Docker")
                return
            except Exception as e:
                print(f"✗ Connection error: {str(e)}")
                return
            
            # Test model
            print(f"\n2. Testing model {model_name}...")
            test_payload = {
                "model": model_name,
                "prompt": "Hello, this is a test."
            }
            
            try:
                async with session.post(
                    f"{ollama_url}/api/embeddings",
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    print(f"Model test status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        if "embedding" in result and result["embedding"]:
                            print(f"✓ Model test successful - embedding length: {len(result['embedding'])}")
                        else:
                            print(f"✗ Model test failed - no embedding data")
                    else:
                        error_text = await response.text()
                        print(f"✗ Model test failed: {error_text}")
            except asyncio.TimeoutError:
                print("✗ Model test timeout")
            except Exception as e:
                print(f"✗ Model test error: {str(e)}")
    
    except Exception as e:
        print(f"✗ General error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_docker_ollama_connection())
