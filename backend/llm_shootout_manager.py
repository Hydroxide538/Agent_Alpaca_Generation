import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import aiohttp
from pathlib import Path

from backend.llm_manager import LLMManager
from backend.improved_alpaca_generator import ImprovedAlpacaGenerator, ExtractedFact, ExtractedConcept
from backend.rag_system import RAGSystem
from backend.manager_scoring_system import ManagerScoringSystem

logger = logging.getLogger(__name__)

class LLMShootoutManager:
    """Manages LLM Shootout competitions with real-time progress tracking"""
    
    def __init__(self, ollama_url: str = "http://host.docker.internal:11434"):
        self.ollama_url = ollama_url
        self.llm_manager = LLMManager()
        self.llm_manager.ollama_base_url = ollama_url
        
        # Initialize components
        self.rag_system = RAGSystem(embedding_model="ollama:bge-m3:latest", reranking_model="ollama:bge-m3:latest")
        self.generator = ImprovedAlpacaGenerator(self.llm_manager, self.rag_system)
        self.manager_scorer = ManagerScoringSystem(self.llm_manager, "ollama:llama3.3:latest")
        
        # Competition state
        self.is_running = False
        self.current_competition = None
        self.progress_callbacks = []
        
        # Evaluation tasks
        self.evaluation_tasks = {
            "fact_extraction": {
                "description": "Extract 5 specific, verifiable facts from the document.",
                "weight": 1.0
            },
            "concept_extraction": {
                "description": "Extract 3 key concepts from the document.",
                "weight": 1.0
            },
            "analytical_qa": {
                "description": "Generate one analytical Q&A pair based on two concepts.",
                "weight": 1.2
            },
        }
    
    def add_progress_callback(self, callback: Callable):
        """Add a callback function for progress updates"""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable):
        """Remove a progress callback"""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    async def _broadcast_progress(self, message_type: str, data: Dict[str, Any]):
        """Broadcast progress updates to all registered callbacks"""
        message = {
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    async def discover_ollama_models(self) -> List[Dict[str, Any]]:
        """Dynamically discover available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status != 200:
                        raise Exception(f"Failed to connect to Ollama: HTTP {response.status}")
                    
                    data = await response.json()
                    models = data.get("models", [])
                    
                    discovered_models = []
                    for model in models:
                        model_name = model["name"]
                        model_info = {
                            "name": model_name,
                            "id": model.get("digest", ""),
                            "size": model.get("size", 0),
                            "modified": model.get("modified_at", ""),
                            "type": self._detect_model_type(model_name, model.get("details", {}))
                        }
                        discovered_models.append(model_info)
                    
                    return discovered_models
                    
        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            return []
    
    def _detect_model_type(self, model_name: str, details: Dict) -> str:
        """Detect if model is text generation or embedding based on name and details"""
        model_name_lower = model_name.lower()
        
        # Check for embedding models
        embedding_indicators = ['embed', 'bge', 'arctic-embed', 'e5-', 'sentence-transformer']
        if any(indicator in model_name_lower for indicator in embedding_indicators):
            return "embedding"
        
        # Check model family from details
        family = details.get("family", "").lower()
        families = details.get("families", [])
        
        if family == "bert" or "bert" in families:
            return "embedding"
        
        # Default to text generation
        return "text_generation"
    
    async def get_available_documents(self) -> List[Dict[str, Any]]:
        """Get list of available documents for evaluation"""
        documents = []
        
        # Check backend/uploads directory
        uploads_dir = Path(__file__).parent / "uploads"
        if uploads_dir.exists():
            for file_path in uploads_dir.glob("*.pdf"):
                documents.append({
                    "id": file_path.name,
                    "name": file_path.stem,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return documents
    
    async def start_shootout(self, document_id: str, selected_models: List[str]) -> Dict[str, Any]:
        """Start an LLM shootout competition"""
        if self.is_running:
            raise Exception("A shootout is already running")
        
        # Validate document
        documents = await self.get_available_documents()
        document = next((d for d in documents if d["id"] == document_id), None)
        if not document:
            raise Exception(f"Document {document_id} not found")
        
        # Discover and validate models
        available_models = await self.discover_ollama_models()
        text_gen_models = [m for m in available_models if m["type"] == "text_generation"]
        
        # Filter to selected models
        competing_models = [m for m in text_gen_models if m["name"] in selected_models]
        if len(competing_models) < 2:
            raise Exception("At least 2 models are required for competition")
        
        # Initialize competition
        self.current_competition = {
            "id": f"shootout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "document": document,
            "models": competing_models,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "results": {}
        }
        
        self.is_running = True
        
        # Start competition in background
        asyncio.create_task(self._run_competition())
        
        await self._broadcast_progress("shootout_started", {
            "competition_id": self.current_competition["id"],
            "document": document["name"],
            "models": [m["name"] for m in competing_models]
        })
        
        return {"status": "started", "competition_id": self.current_competition["id"]}
    
    async def stop_shootout(self) -> Dict[str, Any]:
        """Stop the current shootout competition"""
        if not self.is_running:
            return {"status": "not_running"}
        
        self.is_running = False
        
        if self.current_competition:
            self.current_competition["status"] = "stopped"
            self.current_competition["stopped_at"] = datetime.now().isoformat()
        
        await self._broadcast_progress("shootout_stopped", {
            "competition_id": self.current_competition["id"] if self.current_competition else None
        })
        
        return {"status": "stopped"}
    
    async def get_competition_status(self) -> Dict[str, Any]:
        """Get current competition status"""
        if not self.current_competition:
            return {"status": "no_competition"}
        
        return {
            "status": self.current_competition["status"],
            "competition": self.current_competition
        }
    
    async def _run_competition(self):
        """Run the actual competition"""
        try:
            document_path = self.current_competition["document"]["path"]
            models = self.current_competition["models"]
            
            await self._broadcast_progress("competition_started", {
                "total_models": len(models),
                "total_tasks": len(self.evaluation_tasks)
            })
            
            # Evaluate each model
            for i, model_info in enumerate(models):
                if not self.is_running:
                    break
                
                model_name = model_info["name"]
                model_spec = f"ollama:{model_name}"
                
                await self._broadcast_progress("model_started", {
                    "model": model_name,
                    "progress": (i / len(models)) * 100
                })
                
                # Initialize model results
                self.current_competition["results"][model_name] = {
                    "scores": {},
                    "status": "running",
                    "started_at": datetime.now().isoformat()
                }
                
                try:
                    # Run evaluation tasks for this model
                    await self._evaluate_model(model_spec, document_path)
                    
                    # Mark model as complete
                    self.current_competition["results"][model_name]["status"] = "complete"
                    self.current_competition["results"][model_name]["completed_at"] = datetime.now().isoformat()
                    
                    await self._broadcast_progress("model_completed", {
                        "model": model_name,
                        "scores": self.current_competition["results"][model_name]["scores"]
                    })
                    
                except Exception as e:
                    logger.error(f"Error evaluating model {model_name}: {e}")
                    self.current_competition["results"][model_name]["status"] = "failed"
                    self.current_competition["results"][model_name]["error"] = str(e)
                    
                    await self._broadcast_progress("model_failed", {
                        "model": model_name,
                        "error": str(e)
                    })
            
            # Competition complete
            if self.is_running:
                self.current_competition["status"] = "complete"
                self.current_competition["completed_at"] = datetime.now().isoformat()
                
                await self._broadcast_progress("shootout_completed", {
                    "competition_id": self.current_competition["id"],
                    "results": self.current_competition["results"]
                })
            
        except Exception as e:
            logger.error(f"Error in competition: {e}")
            if self.current_competition:
                self.current_competition["status"] = "error"
                self.current_competition["error"] = str(e)
            
            await self._broadcast_progress("shootout_error", {
                "error": str(e)
            })
        
        finally:
            self.is_running = False
    
    async def _evaluate_model(self, model_spec: str, document_path: str):
        """Evaluate a single model on all tasks"""
        model_name = model_spec.split(":", 1)[1]
        config = {"data_generation_model": model_spec}
        
        for task_name, task_info in self.evaluation_tasks.items():
            if not self.is_running:
                break
            
            await self._broadcast_progress("task_started", {
                "model": model_name,
                "task": task_name,
                "description": task_info["description"]
            })
            
            try:
                # Run the task
                if task_name == "fact_extraction":
                    result = await self._robust_extract_facts(document_path, config)
                    score = await self._score_fact_extraction(result, document_path)
                elif task_name == "concept_extraction":
                    result = await self._robust_extract_concepts(document_path, config)
                    score = await self._score_concept_extraction(result, document_path)
                elif task_name == "analytical_qa":
                    result = await self._robust_generate_analytical_qa(document_path, config)
                    score = await self._score_qa_generation(result)
                else:
                    score = 0.0
                
                # Store score
                self.current_competition["results"][model_name]["scores"][task_name] = score
                self.llm_manager.update_model_performance(model_spec, task_name, score)
                
                await self._broadcast_progress("task_completed", {
                    "model": model_name,
                    "task": task_name,
                    "score": score
                })
                
            except Exception as e:
                logger.error(f"Error in task {task_name} for model {model_name}: {e}")
                self.current_competition["results"][model_name]["scores"][task_name] = 0.0
                
                await self._broadcast_progress("task_failed", {
                    "model": model_name,
                    "task": task_name,
                    "error": str(e)
                })
    
    async def _robust_extract_facts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedFact]:
        """Robust fact extraction with enhanced error handling"""
        try:
            result = await self.generator._extract_structured_facts(doc_path, config)
            if not result:
                result = await self._fallback_fact_extraction(doc_path, config)
            return result
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return await self._fallback_fact_extraction(doc_path, config)
    
    async def _robust_extract_concepts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedConcept]:
        """Robust concept extraction with enhanced error handling"""
        try:
            result = await self.generator._extract_structured_concepts(doc_path, config)
            if not result:
                result = await self._fallback_concept_extraction(doc_path, config)
            return result
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return await self._fallback_concept_extraction(doc_path, config)
    
    async def _robust_generate_analytical_qa(self, doc_path: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Robust Q&A generation with enhanced error handling"""
        try:
            concepts = await self._robust_extract_concepts(doc_path, config)
            if len(concepts) < 2:
                return []
            
            result = await self.generator._generate_analytical_qa_pairs(concepts[:2], config)
            if not result:
                result = await self._fallback_qa_generation(concepts[:2], config)
            return result
        except Exception as e:
            logger.error(f"Q&A generation failed: {e}")
            return []
    
    async def _fallback_fact_extraction(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedFact]:
        """Fallback method for fact extraction using simpler prompts"""
        try:
            model_spec = config["data_generation_model"]
            document_content = self.generator._read_document_content(doc_path)
            
            prompt = f"""
Extract 5 specific facts from the document. Return only a JSON array like this:
[
  {{"content": "fact 1", "context": "context", "fact_type": "general", "confidence": 0.8}},
  {{"content": "fact 2", "context": "context", "fact_type": "general", "confidence": 0.8}}
]

Document content: {document_content[:2000]}...
"""
            
            response = await self.llm_manager.generate_text(
                model_spec, prompt, 
                type('Config', (), {'openai_api_key': None, 'ollama_url': self.ollama_url})()
            )
            
            # Clean and parse response
            cleaned_response = self._clean_model_response(response)
            facts_data = json.loads(cleaned_response)
            
            facts = []
            for fact_data in facts_data[:5]:
                fact = ExtractedFact(
                    content=fact_data.get("content", ""),
                    context=fact_data.get("context", ""),
                    fact_type=fact_data.get("fact_type", "general"),
                    confidence=fact_data.get("confidence", 0.5)
                )
                facts.append(fact)
            return facts
        except Exception as e:
            logger.error(f"Fallback fact extraction failed: {e}")
            return []
    
    async def _fallback_concept_extraction(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedConcept]:
        """Fallback method for concept extraction using simpler prompts"""
        try:
            model_spec = config["data_generation_model"]
            document_content = self.generator._read_document_content(doc_path)
            
            prompt = f"""
Extract 3 key concepts from the document. Return only a JSON array like this:
[
  {{"name": "Concept 1", "definition": "definition", "examples": ["example1"], "relationships": ["related concept"], "domain": "field", "confidence": "high"}},
  {{"name": "Concept 2", "definition": "definition", "examples": ["example1"], "relationships": ["related concept"], "domain": "field", "confidence": "high"}}
]

Document content: {document_content[:2000]}...
"""
            
            response = await self.llm_manager.generate_text(
                model_spec, prompt,
                type('Config', (), {'openai_api_key': None, 'ollama_url': self.ollama_url})()
            )
            
            # Clean and parse response
            cleaned_response = self._clean_model_response(response)
            concepts_data = json.loads(cleaned_response)
            
            concepts = []
            for concept_data in concepts_data[:3]:
                concept = ExtractedConcept(
                    name=concept_data.get("name", ""),
                    definition=concept_data.get("definition", ""),
                    examples=concept_data.get("examples", []),
                    relationships=concept_data.get("relationships", []),
                    domain=concept_data.get("domain", ""),
                    confidence=concept_data.get("confidence", "medium")
                )
                concepts.append(concept)
            return concepts
        except Exception as e:
            logger.error(f"Fallback concept extraction failed: {e}")
            return []
    
    async def _fallback_qa_generation(self, concepts: List[ExtractedConcept], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Fallback method for Q&A generation"""
        try:
            model_spec = config["data_generation_model"]
            concept_names = [c.name for c in concepts[:2]]
            
            prompt = f"""
Create 1 analytical question and answer based on these concepts: {', '.join(concept_names)}

Return only a JSON array like this:
[
  {{"instruction": "Compare and contrast {concept_names[0]} and {concept_names[1]}", "output": "detailed analytical answer"}}
]
"""
            
            response = await self.llm_manager.generate_text(
                model_spec, prompt,
                type('Config', (), {'openai_api_key': None, 'ollama_url': self.ollama_url})()
            )
            
            # Clean and parse response
            cleaned_response = self._clean_model_response(response)
            qa_data = json.loads(cleaned_response)
            return qa_data[:1]
        except Exception as e:
            logger.error(f"Fallback Q&A generation failed: {e}")
            return []
    
    def _clean_model_response(self, response: str) -> str:
        """Clean model response to extract valid JSON"""
        if not response:
            return "[]"
        
        import re
        
        # Remove <think> tags and content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON array or object
        json_match = re.search(r'(\[.*\]|\{.*\})', response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        return "[]"
    
    async def _score_fact_extraction(self, result: List[ExtractedFact], document_path: str) -> float:
        """Score fact extraction with manager LLM and fallback"""
        if not result:
            return 0.0
        
        try:
            document_content = self.generator._read_document_content(document_path)
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_fact_extraction(result, document_content, config)
            return score
        except Exception as e:
            logger.warning(f"Manager scoring failed, using fallback: {e}")
            return self._fallback_score_fact_extraction(result)
    
    async def _score_concept_extraction(self, result: List[ExtractedConcept], document_path: str) -> float:
        """Score concept extraction with manager LLM and fallback"""
        if not result:
            return 0.0
        
        try:
            document_content = self.generator._read_document_content(document_path)
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_concept_extraction(result, document_content, config)
            return score
        except Exception as e:
            logger.warning(f"Manager scoring failed, using fallback: {e}")
            return self._fallback_score_concept_extraction(result)
    
    async def _score_qa_generation(self, result: List[Dict[str, str]]) -> float:
        """Score Q&A generation with manager LLM and fallback"""
        if not result:
            return 0.0
        
        try:
            # For Q&A scoring, we need concepts for context
            # This is a simplified version - in practice, we'd pass the concepts used
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            # Simplified scoring without concepts for now
            return self._fallback_score_qa_generation(result)
        except Exception as e:
            logger.warning(f"Manager scoring failed, using fallback: {e}")
            return self._fallback_score_qa_generation(result)
    
    def _fallback_score_fact_extraction(self, result: List[ExtractedFact]) -> float:
        """Fallback scoring for fact extraction"""
        if not result:
            return 0.0
        
        total_score = 0.0
        for fact in result:
            score = 0.0
            
            # Content quality (0-0.4)
            if fact.content and len(fact.content.strip()) >= 10:
                score += 0.4
            elif fact.content and len(fact.content.strip()) >= 5:
                score += 0.2
            
            # Context quality (0-0.3)
            if fact.context and len(fact.context.strip()) >= 10:
                score += 0.3
            elif fact.context and len(fact.context.strip()) >= 5:
                score += 0.15
            
            # Confidence (0-0.3)
            try:
                confidence = float(fact.confidence) if isinstance(fact.confidence, (int, float)) else 0.5
                score += confidence * 0.3
            except:
                score += 0.15
            
            total_score += score
        
        return min(total_score / len(result), 1.0)
    
    def _fallback_score_concept_extraction(self, result: List[ExtractedConcept]) -> float:
        """Fallback scoring for concept extraction"""
        if not result:
            return 0.0
        
        total_score = 0.0
        for concept in result:
            score = 0.0
            
            # Name quality (0-0.2)
            if concept.name and len(concept.name.strip()) >= 3:
                score += 0.2
            
            # Definition quality (0-0.4)
            if concept.definition and len(concept.definition.strip()) >= 20:
                score += 0.4
            elif concept.definition and len(concept.definition.strip()) >= 10:
                score += 0.2
            
            # Examples (0-0.2)
            if concept.examples and len(concept.examples) > 0:
                score += 0.2
            
            # Relationships (0-0.2)
            if concept.relationships and len(concept.relationships) > 0:
                score += 0.2
            
            total_score += score
        
        return min(total_score / len(result), 1.0)
    
    def _fallback_score_qa_generation(self, result: List[Dict[str, str]]) -> float:
        """Fallback scoring for Q&A generation"""
        if not result:
            return 0.0
        
        total_score = 0.0
        for qa_pair in result:
            score = 0.0
            
            instruction = qa_pair.get("instruction", "")
            output = qa_pair.get("output", "")
            
            # Instruction quality (0-0.5)
            if instruction and len(instruction.strip()) >= 10:
                score += 0.5
            elif instruction and len(instruction.strip()) >= 5:
                score += 0.25
            
            # Output quality (0-0.5)
            if output and len(output.strip()) >= 50:
                score += 0.5
            elif output and len(output.strip()) >= 20:
                score += 0.25
            
            total_score += score
        
        return min(total_score / len(result), 1.0)
