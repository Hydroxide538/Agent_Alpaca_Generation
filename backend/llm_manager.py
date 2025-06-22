import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any
import openai
from backend.models import WorkflowConfig, TestResult

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM connections and testing for both OpenAI and Ollama"""
    
    def __init__(self):
        self.openai_client = None
        self.ollama_base_url = "http://host.docker.internal:11434"  # Default for Docker environment
    
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
                message="OpenAI API key not configured. Please provide a valid OpenAI API key to use OpenAI models.",
                error="No API key provided for OpenAI model"
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
            # Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # First check if model exists
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status != 200:
                        return TestResult(
                            success=False,
                            message="Cannot connect to Ollama server",
                            error=f"HTTP {response.status}"
                        )
                    
                    models_data = await response.json()
                    available_models = [model["name"] for model in models_data.get("models", [])]
                    available_model_names = [model.split(":")[0] for model in available_models]
                    
                    # Check if the model name (without tag) exists
                    model_base_name = model_name.split(":")[0]
                    if model_base_name not in available_model_names:
                        return TestResult(
                            success=False,
                            message=f"Model {model_name} not found in Ollama. Available models: {', '.join(available_model_names)}",
                            error="Model not found"
                        )
                    
                    # Find the actual model name to use (prefer exact match, fallback to base name)
                    actual_model_name = model_name
                    if model_name not in available_models:
                        # If the exact model name isn't found, use just the base name
                        actual_model_name = model_base_name
                    
                    # Get model details to determine if it's an embedding model
                    model_details = None
                    for model in models_data.get("models", []):
                        if model["name"] == actual_model_name:
                            model_details = model.get("details", {})
                            break
                    
                    # Check if this is an embedding model (BERT family or embedding-specific models)
                    is_embedding_model = False
                    if model_details:
                        family = model_details.get("family", "").lower()
                        families = model_details.get("families", [])
                        if family == "bert" or "bert" in families or "bge" in model_base_name.lower() or "embed" in model_base_name.lower():
                            is_embedding_model = True
                
                # Test the model based on its type
                if is_embedding_model:
                    # Test embedding model
                    test_payload = {
                        "model": actual_model_name,
                        "prompt": "Hello, this is a test."
                    }
                    
                    async with session.post(
                        f"{self.ollama_base_url}/api/embeddings",
                        json=test_payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "embedding" in result and result["embedding"]:
                                return TestResult(
                                    success=True,
                                    message=f"Ollama embedding model {actual_model_name} is working correctly"
                                )
                            else:
                                return TestResult(
                                    success=False,
                                    message=f"Ollama embedding model {actual_model_name} returned invalid response",
                                    error="No embedding data in response"
                                )
                        else:
                            error_text = await response.text()
                            return TestResult(
                                success=False,
                                message=f"Ollama embedding model {actual_model_name} test failed",
                                error=error_text
                            )
                else:
                    # Test text generation model
                    test_payload = {
                        "model": actual_model_name,
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
                                message=f"Ollama model {actual_model_name} is working correctly"
                            )
                        else:
                            error_text = await response.text()
                            return TestResult(
                                success=False,
                                message=f"Ollama model {actual_model_name} test failed",
                                error=error_text
                            )
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout testing Ollama model {model_name}")
            return TestResult(
                success=False,
                message=f"Timeout testing Ollama model {model_name}",
                error="Connection timeout - check if Ollama is accessible from Docker"
            )
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error testing Ollama model {model_name}: {str(e)}")
            return TestResult(
                success=False,
                message=f"Cannot connect to Ollama server at {self.ollama_base_url}",
                error=f"Connection error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error testing Ollama model {model_name}: {str(e)}")
            return TestResult(
                success=False,
                message=f"Failed to test Ollama model {model_name}: {str(e)}",
                error=str(e)
            )
    
    async def _resolve_ollama_model_name(self, model_name: str, ollama_url: str) -> str:
        """Resolve the actual Ollama model name to use"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/tags") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [model["name"] for model in models_data.get("models", [])]
                        
                        # If exact model name exists, use it
                        if model_name in available_models:
                            return model_name
                        
                        # Otherwise, use just the base name (without tag)
                        model_base_name = model_name.split(":")[0]
                        return model_base_name
                    else:
                        # Fallback to base name if can't connect
                        return model_name.split(":")[0]
        except Exception as e:
            logger.error(f"Error resolving Ollama model name {model_name}: {str(e)}")
            # Fallback to base name
            return model_name.split(":")[0]

    async def get_ollama_models(self, ollama_url: str = None) -> List[str]:
        """Get list of available Ollama models"""
        url = ollama_url or self.ollama_base_url
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data.get("models", [])]
                    else:
                        logger.error(f"Failed to get Ollama models: HTTP {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {str(e)}")
            return []
    
    async def get_available_models(self, ollama_url: str = None) -> List[str]:
        """Get list of available models (alias for get_ollama_models for compatibility)"""
        return await self.get_ollama_models(ollama_url)
    
    async def pull_ollama_model(self, model_name: str, ollama_url: str = None) -> bool:
        """Pull an Ollama model"""
        url = ollama_url or self.ollama_base_url
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": model_name}
                async with session.post(
                    f"{url}/api/pull",
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
                # Resolve the actual model name
                actual_model_name = await self._resolve_ollama_model_name(model_name, config.ollama_url)
                
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": actual_model_name,
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
                # Resolve the actual model name
                actual_model_name = await self._resolve_ollama_model_name(model_name, config.ollama_url)
                
                embeddings = []
                async with aiohttp.ClientSession() as session:
                    for text in texts:
                        payload = {
                            "model": actual_model_name,
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
    
    async def generate_response(self, model_spec: str, prompt: str, config: Dict[str, Any]) -> str:
        """Generate response using specified model (alias for generate_text with dict config)"""
        try:
            # Convert dict config to WorkflowConfig-like object for compatibility
            class ConfigWrapper:
                def __init__(self, config_dict):
                    self.openai_api_key = config_dict.get("openai_api_key")
                    self.ollama_url = config_dict.get("ollama_url", self.ollama_base_url)
            
            config_obj = ConfigWrapper(config)
            return await self.generate_text(model_spec, prompt, config_obj)
            
        except Exception as e:
            logger.error(f"Error generating response with {model_spec}: {str(e)}")
            raise
