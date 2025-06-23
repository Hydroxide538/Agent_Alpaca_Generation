"""
Manager Agent for intelligent LLM selection based on performance data and strategy
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

from backend.llm_manager import LLMManager
from backend.manager_scoring_system import ManagerScoringSystem

logger = logging.getLogger(__name__)

class ManagerAgent:
    """
    Intelligent agent that selects optimal LLMs for different tasks based on:
    - Historical performance data from LLM shootouts
    - Selection strategy (performance, balanced, speed, quality)
    - Task requirements and context
    """
    
    def __init__(self, manager_model_spec: str, llm_manager: LLMManager):
        self.manager_model_spec = manager_model_spec
        self.llm_manager = llm_manager
        self.manager_scorer = ManagerScoringSystem(llm_manager, manager_model_spec)
        
        # Load performance data
        self.performance_data = self._load_performance_data()
        
        # Task-specific requirements
        self.task_requirements = {
            "fact_extraction": {
                "priority": "accuracy",
                "min_score": 0.7,
                "preferred_models": ["llama", "mistral", "qwen"]
            },
            "concept_extraction": {
                "priority": "comprehension",
                "min_score": 0.6,
                "preferred_models": ["llama", "gemma", "qwen"]
            },
            "analytical_qa": {
                "priority": "reasoning",
                "min_score": 0.8,
                "preferred_models": ["llama", "mistral", "codellama"]
            },
            "data_generation": {
                "priority": "creativity",
                "min_score": 0.6,
                "preferred_models": ["llama", "mistral", "gemma"]
            },
            "embedding": {
                "priority": "semantic_understanding",
                "min_score": 0.5,
                "preferred_models": ["bge", "nomic", "e5"]
            }
        }
    
    def _load_performance_data(self) -> Dict[str, Any]:
        """Load historical performance data from LLM shootouts"""
        try:
            performance_file = Path(__file__).parent / "llm_performance_scores.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    return json.load(f)
            else:
                logger.info("No performance data file found, starting with empty data")
                return {}
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            return {}
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            performance_file = Path(__file__).parent / "llm_performance_scores.json"
            with open(performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    async def select_optimal_llm(self, task_type: str, selection_strategy: str = "performance_based", 
                                available_models: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Select the optimal LLM for a given task based on strategy and performance data
        
        Args:
            task_type: Type of task (fact_extraction, concept_extraction, etc.)
            selection_strategy: Strategy for selection (performance_based, balanced, speed_optimized, quality_focused)
            available_models: List of available models to choose from
            
        Returns:
            Tuple of (selected_model_spec, selection_reasoning)
        """
        logger.info(f"Selecting optimal LLM for task: {task_type}, strategy: {selection_strategy}")
        
        # Get available models if not provided
        if available_models is None:
            available_models = await self._get_available_models()
        
        if not available_models:
            raise Exception("No available models found")
        
        # Filter models based on task requirements
        suitable_models = self._filter_suitable_models(task_type, available_models)
        
        if not suitable_models:
            logger.warning(f"No suitable models found for {task_type}, using all available models")
            suitable_models = available_models
        
        # Apply selection strategy
        selected_model, reasoning = await self._apply_selection_strategy(
            task_type, selection_strategy, suitable_models
        )
        
        logger.info(f"Selected model: {selected_model} for task: {task_type}")
        logger.info(f"Selection reasoning: {reasoning['summary']}")
        
        return selected_model, reasoning
    
    async def _get_available_models(self) -> List[str]:
        """Get list of available models from LLM manager"""
        try:
            # Get Ollama models
            ollama_models = await self.llm_manager.get_ollama_models()
            model_specs = [f"ollama:{model}" for model in ollama_models]
            
            # Add OpenAI models if API key is available
            if hasattr(self.llm_manager, 'openai_api_key') and self.llm_manager.openai_api_key:
                openai_models = [
                    "openai:gpt-4",
                    "openai:gpt-4-turbo", 
                    "openai:gpt-3.5-turbo"
                ]
                model_specs.extend(openai_models)
            
            return model_specs
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def _filter_suitable_models(self, task_type: str, available_models: List[str]) -> List[str]:
        """Filter models based on task requirements"""
        if task_type not in self.task_requirements:
            return available_models
        
        requirements = self.task_requirements[task_type]
        preferred_models = requirements.get("preferred_models", [])
        
        # Filter models that match preferred patterns
        suitable_models = []
        for model_spec in available_models:
            model_name = model_spec.split(":", 1)[1].lower() if ":" in model_spec else model_spec.lower()
            
            # Check if model matches any preferred pattern
            for preferred in preferred_models:
                if preferred.lower() in model_name:
                    suitable_models.append(model_spec)
                    break
        
        return suitable_models
    
    async def _apply_selection_strategy(self, task_type: str, strategy: str, 
                                      suitable_models: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Apply the specified selection strategy to choose the best model"""
        
        if strategy == "performance_based":
            return await self._select_by_performance(task_type, suitable_models)
        elif strategy == "balanced":
            return await self._select_balanced(task_type, suitable_models)
        elif strategy == "speed_optimized":
            return await self._select_by_speed(task_type, suitable_models)
        elif strategy == "quality_focused":
            return await self._select_by_quality(task_type, suitable_models)
        else:
            logger.warning(f"Unknown strategy: {strategy}, defaulting to performance_based")
            return await self._select_by_performance(task_type, suitable_models)
    
    async def _select_by_performance(self, task_type: str, models: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select model based purely on historical performance scores"""
        best_model = None
        best_score = -1
        model_scores = {}
        
        for model_spec in models:
            score = self._get_model_performance_score(model_spec, task_type)
            model_scores[model_spec] = score
            
            if score > best_score:
                best_score = score
                best_model = model_spec
        
        # Fallback to first model if no performance data
        if best_model is None:
            best_model = models[0]
            best_score = 0.5
        
        reasoning = {
            "strategy": "performance_based",
            "selected_model": best_model,
            "score": best_score,
            "all_scores": model_scores,
            "summary": f"Selected {best_model} with performance score {best_score:.3f} for {task_type}"
        }
        
        return best_model, reasoning
    
    async def _select_balanced(self, task_type: str, models: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select model balancing performance and speed"""
        best_model = None
        best_combined_score = -1
        model_analysis = {}
        
        for model_spec in models:
            performance_score = self._get_model_performance_score(model_spec, task_type)
            speed_score = self._estimate_speed_score(model_spec)
            
            # Weighted combination: 60% performance, 40% speed
            combined_score = (performance_score * 0.6) + (speed_score * 0.4)
            
            model_analysis[model_spec] = {
                "performance": performance_score,
                "speed": speed_score,
                "combined": combined_score
            }
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_model = model_spec
        
        # Fallback
        if best_model is None:
            best_model = models[0]
            best_combined_score = 0.5
        
        reasoning = {
            "strategy": "balanced",
            "selected_model": best_model,
            "combined_score": best_combined_score,
            "analysis": model_analysis,
            "summary": f"Selected {best_model} with balanced score {best_combined_score:.3f} (60% performance, 40% speed)"
        }
        
        return best_model, reasoning
    
    async def _select_by_speed(self, task_type: str, models: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select model optimized for speed"""
        best_model = None
        best_speed_score = -1
        speed_scores = {}
        
        for model_spec in models:
            speed_score = self._estimate_speed_score(model_spec)
            speed_scores[model_spec] = speed_score
            
            if speed_score > best_speed_score:
                best_speed_score = speed_score
                best_model = model_spec
        
        # Fallback
        if best_model is None:
            best_model = models[0]
            best_speed_score = 0.5
        
        reasoning = {
            "strategy": "speed_optimized",
            "selected_model": best_model,
            "speed_score": best_speed_score,
            "all_speeds": speed_scores,
            "summary": f"Selected {best_model} with speed score {best_speed_score:.3f} for fastest execution"
        }
        
        return best_model, reasoning
    
    async def _select_by_quality(self, task_type: str, models: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Select model focused on highest quality output"""
        # For quality-focused, we prioritize larger models and those with high accuracy
        best_model = None
        best_quality_score = -1
        quality_analysis = {}
        
        for model_spec in models:
            performance_score = self._get_model_performance_score(model_spec, task_type)
            size_bonus = self._get_model_size_bonus(model_spec)
            
            # Quality score emphasizes performance and model size
            quality_score = (performance_score * 0.8) + (size_bonus * 0.2)
            
            quality_analysis[model_spec] = {
                "performance": performance_score,
                "size_bonus": size_bonus,
                "quality_score": quality_score
            }
            
            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_model = model_spec
        
        # Fallback
        if best_model is None:
            best_model = models[0]
            best_quality_score = 0.5
        
        reasoning = {
            "strategy": "quality_focused",
            "selected_model": best_model,
            "quality_score": best_quality_score,
            "analysis": quality_analysis,
            "summary": f"Selected {best_model} with quality score {best_quality_score:.3f} for highest output quality"
        }
        
        return best_model, reasoning
    
    def _get_model_performance_score(self, model_spec: str, task_type: str) -> float:
        """Get historical performance score for a model on a specific task"""
        if model_spec not in self.performance_data:
            return 0.5  # Default neutral score
        
        model_data = self.performance_data[model_spec]
        
        # Get task-specific score
        if task_type in model_data.get("task_scores", {}):
            return model_data["task_scores"][task_type]
        
        # Fallback to overall score
        return model_data.get("overall_score", 0.5)
    
    def _estimate_speed_score(self, model_spec: str) -> float:
        """Estimate speed score based on model characteristics"""
        model_name = model_spec.split(":", 1)[1].lower() if ":" in model_spec else model_spec.lower()
        
        # OpenAI models are generally fast due to API optimization
        if model_spec.startswith("openai:"):
            if "gpt-3.5" in model_name:
                return 0.9
            elif "gpt-4" in model_name:
                return 0.7
            else:
                return 0.8
        
        # Ollama model speed estimation based on size indicators
        if any(size in model_name for size in ["0.5b", "1b", "1.8b"]):
            return 0.95  # Very small models
        elif any(size in model_name for size in ["2b", "3b"]):
            return 0.85  # Small models
        elif any(size in model_name for size in ["7b", "8b"]):
            return 0.6   # Medium models
        elif any(size in model_name for size in ["13b", "14b", "15b"]):
            return 0.4   # Large models
        elif any(size in model_name for size in ["30b", "34b", "70b"]):
            return 0.2   # Very large models
        
        # Default based on model family
        if "phi" in model_name or "gemma" in model_name:
            return 0.8  # Generally faster models
        elif "llama" in model_name or "mistral" in model_name:
            return 0.6  # Medium speed
        else:
            return 0.5  # Unknown, neutral score
    
    def _get_model_size_bonus(self, model_spec: str) -> float:
        """Get bonus score based on model size (larger models often produce higher quality)"""
        model_name = model_spec.split(":", 1)[1].lower() if ":" in model_spec else model_spec.lower()
        
        # OpenAI models
        if model_spec.startswith("openai:"):
            if "gpt-4" in model_name:
                return 0.9
            elif "gpt-3.5" in model_name:
                return 0.7
            else:
                return 0.8
        
        # Size-based scoring for Ollama models
        if any(size in model_name for size in ["70b", "65b"]):
            return 1.0
        elif any(size in model_name for size in ["30b", "34b"]):
            return 0.9
        elif any(size in model_name for size in ["13b", "14b", "15b"]):
            return 0.8
        elif any(size in model_name for size in ["7b", "8b"]):
            return 0.7
        elif any(size in model_name for size in ["3b", "2b"]):
            return 0.5
        elif any(size in model_name for size in ["1b", "0.5b"]):
            return 0.3
        else:
            return 0.6  # Default for unknown sizes
    
    def update_performance_data(self, model_spec: str, task_type: str, score: float, 
                               additional_metrics: Optional[Dict[str, Any]] = None):
        """Update performance data with new results"""
        if model_spec not in self.performance_data:
            self.performance_data[model_spec] = {
                "task_scores": {},
                "overall_score": 0.0,
                "total_evaluations": 0,
                "last_updated": datetime.now().isoformat()
            }
        
        model_data = self.performance_data[model_spec]
        
        # Update task-specific score
        model_data["task_scores"][task_type] = score
        
        # Update overall score (average of all task scores)
        task_scores = list(model_data["task_scores"].values())
        model_data["overall_score"] = sum(task_scores) / len(task_scores)
        
        # Update metadata
        model_data["total_evaluations"] += 1
        model_data["last_updated"] = datetime.now().isoformat()
        
        # Add additional metrics if provided
        if additional_metrics:
            if "metrics" not in model_data:
                model_data["metrics"] = {}
            model_data["metrics"].update(additional_metrics)
        
        # Save updated data
        self._save_performance_data()
        
        logger.info(f"Updated performance data for {model_spec}: {task_type} = {score:.3f}")
    
    async def get_model_recommendations(self, task_type: str) -> Dict[str, Any]:
        """Get model recommendations for a specific task"""
        available_models = await self._get_available_models()
        
        if not available_models:
            return {"error": "No available models found"}
        
        recommendations = {}
        
        # Get recommendations for each strategy
        for strategy in ["performance_based", "balanced", "speed_optimized", "quality_focused"]:
            try:
                selected_model, reasoning = await self._apply_selection_strategy(
                    task_type, strategy, available_models
                )
                recommendations[strategy] = {
                    "model": selected_model,
                    "reasoning": reasoning
                }
            except Exception as e:
                logger.error(f"Failed to get recommendation for strategy {strategy}: {e}")
                recommendations[strategy] = {"error": str(e)}
        
        return {
            "task_type": task_type,
            "available_models": available_models,
            "recommendations": recommendations,
            "performance_data_available": len(self.performance_data) > 0
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance data"""
        if not self.performance_data:
            return {"message": "No performance data available"}
        
        summary = {
            "total_models_evaluated": len(self.performance_data),
            "models": {},
            "task_coverage": {},
            "last_updated": None
        }
        
        all_tasks = set()
        latest_update = None
        
        for model_spec, data in self.performance_data.items():
            model_summary = {
                "overall_score": data.get("overall_score", 0.0),
                "task_scores": data.get("task_scores", {}),
                "total_evaluations": data.get("total_evaluations", 0),
                "last_updated": data.get("last_updated")
            }
            
            summary["models"][model_spec] = model_summary
            all_tasks.update(data.get("task_scores", {}).keys())
            
            # Track latest update
            if data.get("last_updated"):
                update_time = datetime.fromisoformat(data["last_updated"])
                if latest_update is None or update_time > latest_update:
                    latest_update = update_time
        
        # Calculate task coverage
        for task in all_tasks:
            models_with_task = sum(1 for data in self.performance_data.values() 
                                 if task in data.get("task_scores", {}))
            summary["task_coverage"][task] = {
                "models_evaluated": models_with_task,
                "coverage_percentage": (models_with_task / len(self.performance_data)) * 100
            }
        
        if latest_update:
            summary["last_updated"] = latest_update.isoformat()
        
        return summary
