import requests
import json

def test_ollama_api():
    """Test the Ollama API endpoint directly"""
    
    # Test 1: Health check
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
        else:
            print(f"Health error: {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Ollama models with localhost
    print("Testing Ollama models with localhost...")
    try:
        response = requests.get("http://localhost:8000/ollama-models?ollama_url=http://localhost:11434")
        print(f"Localhost Ollama status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Localhost response: {json.dumps(data, indent=2)}")
        else:
            print(f"Localhost error: {response.text}")
    except Exception as e:
        print(f"Localhost test failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Ollama models with Docker URL
    print("Testing Ollama models with Docker URL...")
    try:
        response = requests.get("http://localhost:8000/ollama-models?ollama_url=http://host.docker.internal:11434")
        print(f"Docker Ollama status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Docker response: {json.dumps(data, indent=2)}")
        else:
            print(f"Docker error: {response.text}")
    except Exception as e:
        print(f"Docker test failed: {e}")

if __name__ == "__main__":
    test_ollama_api()
