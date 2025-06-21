import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional
import openai
from backend.models import WorkflowConfig, TestResult

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM connections and testing for both OpenAI and Ollama"""
    
    def __init__(self):
        self.openai_client = None
        self.ollama_base_url = "http://localhost:11434"
    
    def setup_openai(self, api_key: str):
        """Setup OpenAI client with API key"""
        if api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=api_key)
    
    async def test_models(self, config: WorkflowConfig) -> Dict[str, TestResult]:
        """Test all configured models"""
        results = {}
        
        # Setup OpenAI if needed
        if config.openai_api_key:
            self.setup_openai(config.openai_api_key)
        
        # Update Ollama URL if provided
        if config.ollama_url:
            self.ollama_base_url = config.ollama_url
        
        # Test data generation model
        if config.data_generation_model:
            results["data_generation"] = await self._test_model(config.data_generation_model)
        
        # Test embedding model
        if config.embedding_model:
            results["embedding"] = await self._test_model(config.embedding_model)
        
        # Test reranking model if provided
        if config.reranking_model:
            results["reranking"] = await self._test_model(config.reranking_model)
        
        return results
    
    async def _test_model(self, model_spec: str) -> TestResult:
        """Test a specific model"""
        try:
            provider, model_name = model_spec.split(":", 1)
            start_time = time.time()
            
            if provider == "openai":
                result = await self._test_openai_model(model_name)
            elif provider == "ollama":
                result = await self._test_ollama_model(model_name)
            else:
                return TestResult(
                    success=False,
                    message=f"Unknown provider: {provider}",
                    error=f"Provider {provider} is not supported"
                )
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            result.response_time = response_time
            return result
            
        except Exception as e:
            logger.error(f"Error testing model {model_spec}: {str(e)}")
            return TestResult(
                success=False,
                message=f"Failed to test model: {str(e)}",
                error=str(e)
            )
    
    async def _test_openai_model(self, model_name: str) -> TestResult:
        """Test OpenAI model"""
        if not self.openai_client:
            return TestResult(
                success=False,
                message="OpenAI API key not configured",
                error="No API key provided"
            )
        
        try:
            if model_name.startswith("text-embedding"):
                # Test embedding model
                response = await self.openai_client.embeddings.create(
                    model=model_name,
                    input="Test embedding"
                )
                return TestResult(
                    success=True,
                    message=f"OpenAI embedding model {model_name} is working correctly"
                )
            else:
                # Test chat model
                response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Hello, this is a test."}],
                    max_tokens=10
                )
                return TestResult(
                    success=True,
                    message=f"OpenAI chat model {model_name} is working correctly"
                )
        
        except Exception as e:
            return TestResult(
                success=False,
                message=f"OpenAI model {model_name} test failed: {str(e)}",
                error=str(e)
            )
    
    async def _test_ollama_model(self, model_name: str) -> TestResult:
        """Test Ollama model"""
        try:
            async with aiohttp.ClientSession() as session:
                # First check if model exists
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status != 200:
                        return TestResult(
                            success=False,
                            message="Cannot connect to Ollama server",
                            error=f"HTTP {response.status}"
                        )
                    
                    models_data = await response.json()
                    available_models = [model["name"].split(":")[0] for model in models_data.get("models", [])]
                    
                    if model_name not in available_models:
                        return TestResult(
                            success=False,
                            message=f"Model {model_name} not found in Ollama. Available models: {', '.join(available_models)}",
                            error="Model not found"
                        )
                
                # Test the model with a simple generation
                test_payload = {
                    "model": model_name,
                    "prompt": "Hello, this is a test.",
                    "stream": False,
                    "options": {
                        "num_predict": 10
                    }
                }
                
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=test_payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return TestResult(
                            success=True,
                            message=f"Ollama model {model_name} is working correctly"
                        )
                    else:
                        error_text = await response.text()
                        return TestResult(
                            success=False,
                            message=f"Ollama model {model_name} test failed",
                            error=error_text
                        )
        
        except Exception as e:
            return TestResult(
                success=False,
                message=f"Failed to test Ollama model {model_name}: {str(e)}",
                error=str(e)
            )
    
    async def get_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    else:
                        logger.error(f"Failed to get Ollama models: HTTP {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {str(e)}")
            return []
    
    async def pull_ollama_model(self, model_name: str) -> bool:
        """Pull an Ollama model"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": model_name}
                async with session.post(
                    f"{self.ollama_base_url}/api/pull",
                    json=payload
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Error pulling Ollama model {model_name}: {str(e)}")
            return False
    
    def get_model_config(self, model_spec: str, config: WorkflowConfig) -> Dict:
        """Get model configuration for CrewAI"""
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if provider == "openai":
                return {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": model_name,
                            "api_key": config.openai_api_key
                        }
                    }
                }
            elif provider == "ollama":
                return {
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": model_name,
                            "base_url": config.ollama_url
                        }
                    }
                }
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error getting model config for {model_spec}: {str(e)}")
            return {}
    
    async def generate_text(self, model_spec: str, prompt: str, config: WorkflowConfig) -> str:
        """Generate text using specified model"""
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if provider == "openai":
                if not self.openai_client:
                    self.setup_openai(config.openai_api_key)
                
                response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            elif provider == "ollama":
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                    
                    async with session.post(
                        f"{config.ollama_url}/api/generate",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("response", "")
                        else:
                            raise Exception(f"Ollama API error: {response.status}")
            
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error generating text with {model_spec}: {str(e)}")
            raise
    
    async def get_embeddings(self, model_spec: str, texts: List[str], config: WorkflowConfig) -> List[List[float]]:
        """Get embeddings using specified model"""
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if provider == "openai":
                if not self.openai_client:
                    self.setup_openai(config.openai_api_key)
                
                response = await self.openai_client.embeddings.create(
                    model=model_name,
                    input=texts
                )
                return [embedding.embedding for embedding in response.data]
            
            elif provider == "ollama":
                embeddings = []
                async with aiohttp.ClientSession() as session:
                    for text in texts:
                        payload = {
                            "model": model_name,
                            "prompt": text
                        }
                        
                        async with session.post(
                            f"{config.ollama_url}/api/embeddings",
                            json=payload
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                embeddings.append(result.get("embedding", []))
                            else:
                                raise Exception(f"Ollama embeddings API error: {response.status}")
                
                return embeddings
            
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error getting embeddings with {model_spec}: {str(e)}")
            raise
