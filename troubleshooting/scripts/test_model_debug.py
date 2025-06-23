import asyncio
import aiohttp
import json

async def test_bge_m3_model():
    """Test the bge-m3:latest model specifically"""
    ollama_url = "http://localhost:11434"
    model_name = "bge-m3:latest"
    
    print(f"Testing model: {model_name}")
    print(f"Ollama URL: {ollama_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # First check if model exists
            print("\n1. Checking available models...")
            async with session.get(f"{ollama_url}/api/tags") as response:
                if response.status != 200:
                    print(f"ERROR: Cannot connect to Ollama server - HTTP {response.status}")
                    return
                
                models_data = await response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                print(f"Available models: {available_models}")
                
                # Check if the model name (without tag) exists
                model_base_name = model_name.split(":")[0]
                available_model_names = [model.split(":")[0] for model in available_models]
                print(f"Model base name: {model_base_name}")
                print(f"Available base names: {available_model_names}")
                
                if model_base_name not in available_model_names:
                    print(f"ERROR: Model {model_name} not found")
                    return
                
                # Find the actual model name to use
                actual_model_name = model_name
                if model_name not in available_models:
                    actual_model_name = model_base_name
                    print(f"Using base name: {actual_model_name}")
                else:
                    print(f"Using exact name: {actual_model_name}")
                
                # Get model details
                print(f"\n2. Getting model details for {actual_model_name}...")
                model_details = None
                for model in models_data.get("models", []):
                    if model["name"] == actual_model_name:
                        model_details = model.get("details", {})
                        print(f"Model details: {json.dumps(model_details, indent=2)}")
                        break
                
                # Check if this is an embedding model
                is_embedding_model = False
                if model_details:
                    family = model_details.get("family", "").lower()
                    families = model_details.get("families", [])
                    print(f"Family: {family}")
                    print(f"Families: {families}")
                    
                    if family == "bert" or "bert" in families or "bge" in model_base_name.lower() or "embed" in model_base_name.lower():
                        is_embedding_model = True
                
                print(f"Is embedding model: {is_embedding_model}")
            
            # Test the model
            print(f"\n3. Testing model {actual_model_name}...")
            if is_embedding_model:
                print("Testing as embedding model...")
                test_payload = {
                    "model": actual_model_name,
                    "prompt": "Hello, this is a test."
                }
                
                async with session.post(
                    f"{ollama_url}/api/embeddings",
                    json=test_payload
                ) as response:
                    print(f"Response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        if "embedding" in result and result["embedding"]:
                            print(f"SUCCESS: Embedding model working correctly")
                            print(f"Embedding length: {len(result['embedding'])}")
                        else:
                            print(f"ERROR: No embedding data in response")
                            print(f"Response: {result}")
                    else:
                        error_text = await response.text()
                        print(f"ERROR: {error_text}")
            else:
                print("Testing as text generation model...")
                test_payload = {
                    "model": actual_model_name,
                    "prompt": "Hello, this is a test.",
                    "stream": False,
                    "options": {
                        "num_predict": 10
                    }
                }
                
                async with session.post(
                    f"{ollama_url}/api/generate",
                    json=test_payload
                ) as response:
                    print(f"Response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        print(f"SUCCESS: Text generation model working correctly")
                        print(f"Response: {result.get('response', 'No response')}")
                    else:
                        error_text = await response.text()
                        print(f"ERROR: {error_text}")
    
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_bge_m3_model())
