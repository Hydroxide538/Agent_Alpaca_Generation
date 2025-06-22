# Backend module initialization
# Make safe_llm_wrapper available for imports

try:
    from .safe_llm_wrapper import SafeLLMFactory, CrewAICompatibleLLM, SafeOllamaLLM
    __all__ = ['SafeLLMFactory', 'CrewAICompatibleLLM', 'SafeOllamaLLM']
except ImportError:
    # If import fails, define empty list
    __all__ = []
