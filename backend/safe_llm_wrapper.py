"""
Safe LLM wrapper for CrewAI that prevents "list index out of range" errors
by using the safe completion methods instead of direct LiteLLM calls
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from crewai import LLM
from backend.llm_manager import LLMManager
from backend.message_validator import MessageValidator

logger = logging.getLogger(__name__)

class SafeOllamaLLM:
    """
    Safe LLM wrapper for Ollama models that prevents "list index out of range" errors
    by using validated message handling and direct Ollama API calls
    """
    
    def __init__(self, model_spec: str, config: Dict[str, Any]):
        self.model_spec = model_spec
        self.config = config
        self.llm_manager = LLMManager()
        
        # Extract provider and model name
        try:
            self.provider, self.model_name = model_spec.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid model specification: {model_spec}. Expected format: 'provider:model_name'")
        
        # Set up configuration
        self.ollama_url = config.get('ollama_url', 'http://host.docker.internal:11434')
        self.openai_api_key = config.get('openai_api_key')
        
        # Initialize the underlying LLM manager
        if self.openai_api_key:
            self.llm_manager.setup_openai(self.openai_api_key)
        
        logger.info(f"SafeOllamaLLM initialized for {model_spec}")
    
    def call(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """
        Main call method that CrewAI will use
        Handles both string prompts and message arrays safely
        """
        try:
            # Convert string to message format if needed
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif not isinstance(messages, list):
                messages = [{"role": "user", "content": str(messages)}]
            
            # Use async method in sync context
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If we're already in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._sync_completion, messages, **kwargs)
                    return future.result()
            else:
                return loop.run_until_complete(self._async_completion(messages, **kwargs))
                
        except Exception as e:
            logger.error(f"SafeOllamaLLM call failed: {str(e)}")
            # Return a safe fallback response
            return "I apologize, but I'm having trouble processing your request due to a technical issue."
    
    def _sync_completion(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Synchronous wrapper for async completion"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._async_completion(messages, **kwargs))
        finally:
            loop.close()
    
    async def _async_completion(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Async completion using safe methods"""
        try:
            if self.provider == "ollama":
                # Use the safe Ollama completion method
                return await self.llm_manager.safe_ollama_completion(
                    self.model_spec, 
                    messages, 
                    self.config
                )
            elif self.provider == "openai":
                # Use OpenAI completion
                if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
                    logger.warning("OpenAI API key not provided, falling back to Ollama")
                    # Fall back to Ollama completion
                    fallback_spec = "ollama:llama3.3:latest"
                    return await self.llm_manager.safe_ollama_completion(
                        fallback_spec, 
                        messages, 
                        self.config
                    )
                
                # Convert messages to prompt for generate_text method
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                
                combined_prompt = "\n".join(prompt_parts)
                
                # Create a config wrapper for compatibility
                class ConfigWrapper:
                    def __init__(self, config_dict):
                        self.openai_api_key = config_dict.get("openai_api_key")
                        self.ollama_url = config_dict.get("ollama_url")
                
                config_obj = ConfigWrapper(self.config)
                return await self.llm_manager.generate_text(self.model_spec, combined_prompt, config_obj)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Async completion failed: {str(e)}")
            # Use fallback completion
            return await self.llm_manager._fallback_ollama_completion(
                self.model_spec, 
                "Please provide a helpful response.", 
                self.config
            )
    
    # CrewAI compatibility methods
    def __call__(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """Make the object callable"""
        return self.call(messages, **kwargs)
    
    @property
    def model(self) -> str:
        """Return model name for CrewAI compatibility"""
        return self.model_spec
    
    def __str__(self) -> str:
        return f"SafeOllamaLLM({self.model_spec})"
    
    def __repr__(self) -> str:
        return self.__str__()


class SafeLLMFactory:
    """Factory for creating safe LLM instances"""
    
    @staticmethod
    def create_safe_llm(model_spec: str, config: Dict[str, Any]) -> Union[SafeOllamaLLM, LLM]:
        """
        Create a safe LLM instance based on the model specification
        
        Args:
            model_spec: Model specification in format "provider:model_name"
            config: Configuration dictionary
            
        Returns:
            SafeOllamaLLM for Ollama models, standard LLM for others
        """
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if provider == "ollama":
                # Use our safe wrapper for Ollama models
                return SafeOllamaLLM(model_spec, config)
            elif provider == "openai":
                # For OpenAI, we can use the standard LLM class as it doesn't have the same issues
                api_key = config.get('openai_api_key')
                if not api_key or api_key == "your_openai_api_key_here":
                    logger.warning(f"OpenAI API key not provided for {model_spec}. Falling back to Ollama model.")
                    # Fall back to Ollama model
                    fallback_spec = "ollama:llama3.3:latest"
                    return SafeOllamaLLM(fallback_spec, config)
                return LLM(
                    model=f"openai/{model_name}",
                    api_key=api_key
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error creating safe LLM for {model_spec}: {str(e)}")
            # Return None to indicate failure
            return None
    
    @staticmethod
    def test_safe_llm(safe_llm: SafeOllamaLLM, test_message: str = "Hello, this is a test.") -> bool:
        """
        Test a safe LLM instance
        
        Args:
            safe_llm: SafeOllamaLLM instance to test
            test_message: Test message to send
            
        Returns:
            True if test successful, False otherwise
        """
        try:
            response = safe_llm.call(test_message)
            return len(response.strip()) > 0
        except Exception as e:
            logger.error(f"Safe LLM test failed: {str(e)}")
            return False


# Compatibility wrapper that mimics CrewAI LLM interface
class CrewAICompatibleLLM(LLM):
    """
    Wrapper that provides CrewAI LLM-compatible interface while using safe completion
    This class inherits from the base LLM class to ensure full compatibility
    """
    
    def __init__(self, model_spec: str, config: Dict[str, Any]):
        self.safe_llm = SafeOllamaLLM(model_spec, config)
        self.model_spec = model_spec
        self.config = config
        
        # Set attributes that CrewAI LLM class expects
        provider, model_name = model_spec.split(":", 1)
        if provider == "ollama":
            model_name_for_llm = f"ollama/{model_name}"
            base_url = config.get('ollama_url', 'http://host.docker.internal:11434')
        else:
            model_name_for_llm = model_spec
            base_url = None
        
        # Initialize the parent LLM class
        try:
            if base_url:
                super().__init__(model=model_name_for_llm, base_url=base_url)
            else:
                super().__init__(model=model_name_for_llm)
        except Exception as e:
            logger.warning(f"Failed to initialize parent LLM class: {str(e)}")
            # Set attributes manually if parent init fails
            self.model = model_name_for_llm
            if base_url:
                self.base_url = base_url
        
        # Disable tool calling to prevent issues
        self.supports_tool_calling = False
        self.tool_calling = False
        
        logger.info(f"CrewAICompatibleLLM initialized with model: {self.model}")
    
    def call(self, messages, **kwargs):
        """CrewAI LLM call method"""
        logger.debug(f"CrewAICompatibleLLM.call() invoked with messages type: {type(messages)}")
        return self.safe_llm.call(messages, **kwargs)
    
    def __call__(self, messages, **kwargs):
        """Make callable"""
        logger.debug(f"CrewAICompatibleLLM.__call__() invoked with messages type: {type(messages)}")
        return self.call(messages, **kwargs)
    
    # Additional methods that CrewAI LLM might expect
    def invoke(self, messages, **kwargs):
        """Alternative invoke method"""
        logger.debug(f"CrewAICompatibleLLM.invoke() invoked with messages type: {type(messages)}")
        return self.call(messages, **kwargs)
    
    def generate(self, messages, **kwargs):
        """Alternative generate method"""
        logger.debug(f"CrewAICompatibleLLM.generate() invoked with messages type: {type(messages)}")
        return self.call(messages, **kwargs)
    
    def chat(self, messages, **kwargs):
        """Alternative chat method"""
        logger.debug(f"CrewAICompatibleLLM.chat() invoked with messages type: {type(messages)}")
        return self.call(messages, **kwargs)
    
    # Properties for CrewAI compatibility
    @property
    def model_name(self):
        """Model name property"""
        return self.model
    
    @property
    def provider(self):
        """Provider property"""
        return self.model_spec.split(":")[0]
    
    def __str__(self):
        return f"CrewAICompatibleLLM({self.model_spec})"
    
    def __repr__(self):
        return self.__str__()
    
    def __getattr__(self, name):
        """Fallback for any missing attributes"""
        logger.debug(f"CrewAICompatibleLLM.__getattr__() called for attribute: {name}")
        # If the attribute doesn't exist, return a safe default or delegate to safe_llm
        if hasattr(self.safe_llm, name):
            return getattr(self.safe_llm, name)
        else:
            # Return a safe default for unknown attributes
            return None
