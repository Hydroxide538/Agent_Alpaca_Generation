import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any
import openai
import yaml
import os
import json
from backend.models import WorkflowConfig, TestResult
from backend.message_validator import MessageValidator

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM connections and testing for both OpenAI and Ollama, and provides dynamic model selection."""
    
    def __init__(self):
        self.openai_client = None
        self.ollama_base_url = "http://host.docker.internal:11434"  # Default for Docker environment
        self.llm_registry = self._load_llm_registry()
        self.llm_performance_scores = self._load_llm_performance_scores()
    
    def _load_llm_registry(self) -> Dict[str, Any]:
        """Load LLM registry from config/llms.yaml"""
        registry_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llms.yaml')
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = yaml.safe_load(f)
                logger.info(f"Loaded LLM registry from {registry_path}")
                return registry
        except FileNotFoundError:
            logger.warning(f"LLM registry file not found at {registry_path}. Returning empty registry.")
            return {"ollama_models": [], "external_models": []}
        except Exception as e:
            logger.error(f"Error loading LLM registry from {registry_path}: {str(e)}")
            return {"ollama_models": [], "external_models": []}

    def _load_llm_performance_scores(self) -> Dict[str, Dict[str, Any]]:
        """Load LLM performance scores from backend/llm_performance_scores.json"""
        scores_path = os.path.join(os.path.dirname(__file__), 'llm_performance_scores.json')
        try:
            if os.path.exists(scores_path):
                with open(scores_path, 'r', encoding='utf-8') as f:
                    scores = json.load(f)
                    logger.info(f"Loaded LLM performance scores from {scores_path}")
                    return scores
            return {}
        except Exception as e:
            logger.error(f"Error loading LLM performance scores from {scores_path}: {str(e)}")
            return {}

    def save_llm_performance_scores(self):
        """Save LLM performance scores to backend/llm_performance_scores.json"""
        scores_path = os.path.join(os.path.dirname(__file__), 'llm_performance_scores.json')
        try:
            with open(scores_path, 'w', encoding='utf-8') as f:
                json.dump(self.llm_performance_scores, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved LLM performance scores to {scores_path}")
        except Exception as e:
            logger.error(f"Error saving LLM performance scores to {scores_path}: {str(e)}")
    
    def setup_openai(self, api_key: str):
        """Setup OpenAI client with API key"""
        if api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=api_key)
    
    def get_model_info(self, model_spec: str) -> Optional[Dict[str, Any]]:
        """Retrieve detailed information about a model from the registry."""
        provider, model_name = model_spec.split(":", 1)
        
        if provider == "ollama":
            for model in self.llm_registry.get("ollama_models", []):
                if model["name"] == model_name:
                    return model
        elif provider in ["openai", "claude"]: # Assuming claude models are listed under external
            for model in self.llm_registry.get("external_models", []):
                if model["name"] == model_spec: # External models use full spec as name
                    return model
        return None

    def get_available_ollama_models_from_registry(self) -> List[Dict[str, Any]]:
        """Get list of Ollama models from the registry."""
        return self.llm_registry.get("ollama_models", [])

    def get_available_external_models_from_registry(self) -> List[Dict[str, Any]]:
        """Get list of external models from the registry."""
        return self.llm_registry.get("external_models", [])

    def get_model_performance(self, model_spec: str, task_type: str) -> Optional[float]:
        """Get the performance score of a model for a specific task type."""
        return self.llm_performance_scores.get(model_spec, {}).get(task_type)

    def update_model_performance(self, model_spec: str, task_type: str, score: float):
        """Update the performance score of a model for a specific task type."""
        if model_spec not in self.llm_performance_scores:
            self.llm_performance_scores[model_spec] = {}
        self.llm_performance_scores[model_spec][task_type] = score
        self.save_llm_performance_scores() # Persist scores immediately

    async def test_models(self, config: WorkflowConfig) -> Dict[str, TestResult]:
        """Test all configured models and all models in the registry."""
        results = {}
        
        # Setup OpenAI if needed
        if config.openai_api_key:
            self.setup_openai(config.openai_api_key)
        
        # Update Ollama URL if provided
        if config.ollama_url:
            self.ollama_base_url = config.ollama_url
        
        # Test models specified in workflow config
        if config.data_generation_model:
            results["data_generation"] = await self._test_model(config.data_generation_model)
        if config.embedding_model:
            results["embedding"] = await self._test_model(config.embedding_model)
        if config.reranking_model:
            results["reranking"] = await self._test_model(config.reranking_model)

        # Test all Ollama models from the registry
        for model_info in self.llm_registry.get("ollama_models", []):
            model_spec = f"ollama:{model_info['name']}"
            if model_spec not in results: # Avoid re-testing already tested models
                results[model_spec] = await self._test_model(model_spec)
        
        # Test all external models from the registry if API keys are available
        for model_info in self.llm_registry.get("external_models", []):
            if model_info["name"].startswith("openai:") and config.openai_api_key:
                if model_info["name"] not in results:
                    results[model_info["name"]] = await self._test_model(model_info["name"])
            # Add similar checks for Claude if its API key is managed
        
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
            elif provider == "claude": # Placeholder for Claude testing
                result = TestResult(success=False, message="Claude testing not yet implemented.")
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
            # Check if it's an embedding model based on common naming conventions or registry info
            model_info = self.get_model_info(f"openai:{model_name}")
            is_embedding_model = model_info and model_info.get("type") == "embedding" or "embedding" in model_name.lower()

            if is_embedding_model:
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
            # Add timeout to prevent hanging - increased for large models like llama3.3
            timeout = aiohttp.ClientTimeout(total=300, connect=30)
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
    
    async def generate_text(self, model_spec: str, prompt: str, config: WorkflowConfig, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate text using specified model, supporting chat messages."""
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if messages is None:
                messages = [{"role": "user", "content": prompt}]
            else:
                # Ensure the prompt is added to messages if it's not already there
                if not any(msg.get("content") == prompt for msg in messages):
                    messages.append({"role": "user", "content": prompt})

            if provider == "openai":
                if not self.openai_client:
                    self.setup_openai(config.openai_api_key)
                
                response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            elif provider == "ollama":
                # Resolve the actual model name
                ollama_url = config.ollama_url if hasattr(config, 'ollama_url') else self.ollama_base_url
                actual_model_name = await self._resolve_ollama_model_name(model_name, ollama_url)
                
                async with aiohttp.ClientSession() as session:
                    # Convert messages to a single prompt for Ollama's /api/generate endpoint
                    # Ollama's /api/generate endpoint primarily takes a single prompt string.
                    # For chat-like interactions, we concatenate messages.
                    combined_prompt = ""
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "system":
                            combined_prompt += f"System: {content}\n"
                        elif role == "user":
                            combined_prompt += f"User: {content}\n"
                        elif role == "assistant":
                            combined_prompt += f"Assistant: {content}\n"
                    
                    payload = {
                        "model": actual_model_name,
                        "prompt": combined_prompt.strip(),
                        "stream": False,
                        "options": {
                            "temperature": 0.7, # Default temperature
                            "num_predict": 1000 # Default max tokens
                        }
                    }
                    
                    async with session.post(
                        f"{ollama_url}/api/generate",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("response", "")
                        else:
                            error_text = await response.text()
                            raise Exception(f"Ollama API error: {response.status} - {error_text}")
            
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
                ollama_url = config.ollama_url if hasattr(config, 'ollama_url') else self.ollama_base_url
                actual_model_name = await self._resolve_ollama_model_name(model_name, ollama_url)
                
                embeddings = []
                async with aiohttp.ClientSession() as session:
                    for text in texts:
                        payload = {
                            "model": actual_model_name,
                            "prompt": text
                        }
                        
                        async with session.post(
                            f"{ollama_url}/api/embeddings",
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
    
    async def safe_ollama_completion(self, model_spec: str, messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """
        Safe wrapper for Ollama completion with message validation
        Prevents "list index out of range" errors in litellm transformations
        """
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if provider != "ollama":
                raise ValueError(f"This method is only for Ollama models, got: {provider}")
            
            # Validate and sanitize messages
            safe_messages = MessageValidator.validate_messages_for_ollama(messages)
            logger.info(f"Validated {len(messages)} messages to {len(safe_messages)} safe messages")
            
            # Resolve the actual model name
            ollama_url = config.get("ollama_url", self.ollama_base_url)
            actual_model_name = await self._resolve_ollama_model_name(model_name, ollama_url)
            
            # Use direct Ollama API instead of litellm to avoid transformation issues
            async with aiohttp.ClientSession() as session:
                # Convert messages to a single prompt for Ollama
                prompt_parts = []
                for msg in safe_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                
                combined_prompt = "\n".join(prompt_parts)
                if not combined_prompt.strip():
                    combined_prompt = "Hello, please respond."
                
                payload = {
                    "model": actual_model_name,
                    "prompt": combined_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1000
                    }
                }
                
                async with session.post(
                    f"{ollama_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                        
        except IndexError as e:
            if "list index out of range" in str(e):
                logger.warning(f"Caught list index out of range error, using fallback: {str(e)}")
                # Fallback to simple prompt
                return await self._fallback_ollama_completion(model_spec, "Please respond to the user's request.", config)
            raise
        except Exception as e:
            logger.error(f"Error in safe_ollama_completion: {str(e)}")
            raise
    
    async def _fallback_ollama_completion(self, model_spec: str, simple_prompt: str, config: Dict[str, Any]) -> str:
        """Fallback completion method with minimal message structure"""
        try:
            provider, model_name = model_spec.split(":", 1)
            ollama_url = config.get("ollama_url", self.ollama_base_url)
            actual_model_name = await self._resolve_ollama_model_name(model_name, ollama_url)
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": actual_model_name,
                    "prompt": simple_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                }
                
                async with session.post(
                    f"{ollama_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "I apologize, but I'm having trouble processing your request.")
                    else:
                        return "I apologize, but I'm having trouble processing your request."
                        
        except Exception as e:
            logger.error(f"Fallback completion also failed: {str(e)}")
            return "I apologize, but I'm having trouble processing your request."

    async def generate_response(self, model_spec: str, prompt: str, config: Dict[str, Any], messages: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate response using specified model (alias for generate_text with dict config)"""
        try:
            # Convert dict config to WorkflowConfig-like object for compatibility
            class ConfigWrapper:
                def __init__(self, config_dict, ollama_base_url):
                    self.openai_api_key = config_dict.get("openai_api_key")
                    self.ollama_url = config_dict.get("ollama_url", ollama_base_url)
                    self.ollama_base_url = self.ollama_url  # Add this attribute for compatibility
            
            config_obj = ConfigWrapper(config, self.ollama_base_url)
            return await self.generate_text(model_spec, prompt, config_obj, messages)
            
        except Exception as e:
            logger.error(f"Error generating response with {model_spec}: {str(e)}")
            raise
