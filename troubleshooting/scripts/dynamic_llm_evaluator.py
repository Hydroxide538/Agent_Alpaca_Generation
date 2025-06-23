import asyncio
import os
import sys
import argparse
import json
import re
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.llm_manager import LLMManager
from backend.improved_alpaca_generator import ImprovedAlpacaGenerator, ExtractedFact, ExtractedConcept
from backend.rag_system import RAGSystem
from backend.json_parser_fix import RobustJSONParser
from backend.manager_scoring_system import ManagerScoringSystem

class DynamicLLMEvaluator:
    """Enhanced LLM evaluator with dynamic model detection and robust error handling."""

    def __init__(self, document_path: str, manager_model: str = "ollama:llama3.3:latest", ollama_url: str = "http://host.docker.internal:11434"):
        self.document_path = document_path
        self.ollama_url = ollama_url
        self.llm_manager = LLMManager()
        self.llm_manager.ollama_base_url = ollama_url
        
        # Initialize RAG system
        self.rag_system = RAGSystem(embedding_model="ollama:bge-m3:latest", reranking_model="ollama:bge-m3:latest")
        self.generator = ImprovedAlpacaGenerator(self.llm_manager, self.rag_system)
        self.manager_scorer = ManagerScoringSystem(self.llm_manager, manager_model)
        
        # Enhanced evaluation tasks
        self.evaluation_tasks = self._create_evaluation_tasks()
        self.document_content = self._load_document_content()
        
        # Progress tracking
        self.progress_callback = None
        self.current_model = None
        self.current_task = None
        
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def _update_progress(self, message: str, progress: float = None):
        """Update progress with optional callback"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        if self.progress_callback:
            self.progress_callback({
                'message': message,
                'progress': progress,
                'model': self.current_model,
                'task': self.current_task,
                'timestamp': timestamp
            })
    
    async def discover_ollama_models(self) -> List[Dict[str, Any]]:
        """Dynamically discover available Ollama models"""
        self._update_progress("🔍 Discovering available Ollama models...")
        
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
                    
                    self._update_progress(f"✅ Discovered {len(discovered_models)} models")
                    return discovered_models
                    
        except Exception as e:
            self._update_progress(f"❌ Failed to discover models: {str(e)}")
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
    
    def _load_document_content(self) -> str:
        """Load document content for manager scoring"""
        try:
            return self.generator._read_document_content(self.document_path)
        except Exception as e:
            self._update_progress(f"⚠️ Warning: Could not load document content: {e}")
            return ""

    def _create_evaluation_tasks(self) -> Dict[str, Any]:
        """Create a standardized set of evaluation tasks."""
        return {
            "fact_extraction": {
                "description": "Extract 5 specific, verifiable facts from the document.",
                "function": self._robust_extract_facts,
                "scoring_function": self._manager_score_fact_extraction,
                "weight": 1.0
            },
            "concept_extraction": {
                "description": "Extract 3 key concepts from the document.",
                "function": self._robust_extract_concepts,
                "scoring_function": self._manager_score_concept_extraction,
                "weight": 1.0
            },
            "analytical_qa": {
                "description": "Generate one analytical Q&A pair based on two concepts.",
                "function": self._robust_generate_analytical_qa,
                "scoring_function": self._manager_score_qa_generation,
                "weight": 1.2  # Higher weight for more complex task
            },
        }

    async def _robust_extract_facts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedFact]:
        """Robust fact extraction with enhanced error handling"""
        try:
            self._update_progress(f"  📋 Extracting facts...")
            result = await self.generator._extract_structured_facts(doc_path, config)
            
            if not result:
                self._update_progress(f"  ⚠️ No facts extracted, trying fallback method...")
                result = await self._fallback_fact_extraction(doc_path, config)
            
            self._update_progress(f"  ✅ Extracted {len(result)} facts")
            return result
            
        except Exception as e:
            self._update_progress(f"  ❌ Fact extraction failed: {str(e)}")
            return await self._fallback_fact_extraction(doc_path, config)

    async def _robust_extract_concepts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedConcept]:
        """Robust concept extraction with enhanced error handling"""
        try:
            self._update_progress(f"  🧠 Extracting concepts...")
            result = await self.generator._extract_structured_concepts(doc_path, config)
            
            if not result:
                self._update_progress(f"  ⚠️ No concepts extracted, trying fallback method...")
                result = await self._fallback_concept_extraction(doc_path, config)
            
            self._update_progress(f"  ✅ Extracted {len(result)} concepts")
            return result
            
        except Exception as e:
            self._update_progress(f"  ❌ Concept extraction failed: {str(e)}")
            return await self._fallback_concept_extraction(doc_path, config)

    async def _robust_generate_analytical_qa(self, doc_path: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Robust Q&A generation with enhanced error handling"""
        try:
            self._update_progress(f"  ❓ Generating analytical Q&A...")
            
            # First get concepts
            concepts = await self._robust_extract_concepts(doc_path, config)
            
            if len(concepts) < 2:
                self._update_progress(f"  ⚠️ Insufficient concepts for Q&A generation")
                return []
            
            # Generate Q&A pairs
            result = await self.generator._generate_analytical_qa_pairs(concepts[:2], config)
            
            if not result:
                self._update_progress(f"  ⚠️ No Q&A generated, trying fallback method...")
                result = await self._fallback_qa_generation(concepts[:2], config)
            
            self._update_progress(f"  ✅ Generated {len(result)} Q&A pairs")
            return result
            
        except Exception as e:
            self._update_progress(f"  ❌ Q&A generation failed: {str(e)}")
            return []

    async def _fallback_fact_extraction(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedFact]:
        """Fallback method for fact extraction using simpler prompts"""
        try:
            model_spec = config["data_generation_model"]
            
            # Simple prompt for fact extraction
            prompt = f"""
Extract 5 specific facts from the document. Return only a JSON array like this:
[
  {{"content": "fact 1", "context": "context", "fact_type": "general", "confidence": 0.8}},
  {{"content": "fact 2", "context": "context", "fact_type": "general", "confidence": 0.8}}
]

Document content: {self.document_content[:2000]}...
"""
            
            response = await self.llm_manager.generate_text(
                model_spec, prompt, 
                type('Config', (), {'openai_api_key': None, 'ollama_url': self.ollama_url})()
            )
            
            # Clean response
            cleaned_response = self._clean_model_response(response)
            
            # Parse JSON
            try:
                facts_data = json.loads(cleaned_response)
                facts = []
                for fact_data in facts_data[:5]:  # Limit to 5
                    fact = ExtractedFact(
                        content=fact_data.get("content", ""),
                        context=fact_data.get("context", ""),
                        fact_type=fact_data.get("fact_type", "general"),
                        confidence=fact_data.get("confidence", 0.5)
                    )
                    facts.append(fact)
                return facts
            except:
                return []
                
        except Exception as e:
            self._update_progress(f"  ❌ Fallback fact extraction failed: {str(e)}")
            return []

    async def _fallback_concept_extraction(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedConcept]:
        """Fallback method for concept extraction using simpler prompts"""
        try:
            model_spec = config["data_generation_model"]
            
            # Simple prompt for concept extraction
            prompt = f"""
Extract 3 key concepts from the document. Return only a JSON array like this:
[
  {{"name": "Concept 1", "definition": "definition", "examples": ["example1"], "relationships": ["related concept"], "domain": "field", "confidence": "high"}},
  {{"name": "Concept 2", "definition": "definition", "examples": ["example1"], "relationships": ["related concept"], "domain": "field", "confidence": "high"}}
]

Document content: {self.document_content[:2000]}...
"""
            
            response = await self.llm_manager.generate_text(
                model_spec, prompt,
                type('Config', (), {'openai_api_key': None, 'ollama_url': self.ollama_url})()
            )
            
            # Clean response
            cleaned_response = self._clean_model_response(response)
            
            # Parse JSON
            try:
                concepts_data = json.loads(cleaned_response)
                concepts = []
                for concept_data in concepts_data[:3]:  # Limit to 3
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
            except:
                return []
                
        except Exception as e:
            self._update_progress(f"  ❌ Fallback concept extraction failed: {str(e)}")
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
            
            # Clean response
            cleaned_response = self._clean_model_response(response)
            
            # Parse JSON
            try:
                qa_data = json.loads(cleaned_response)
                return qa_data[:1]  # Return only 1 Q&A pair
            except:
                return []
                
        except Exception as e:
            self._update_progress(f"  ❌ Fallback Q&A generation failed: {str(e)}")
            return []

    def _clean_model_response(self, response: str) -> str:
        """Clean model response to extract valid JSON"""
        if not response:
            return "[]"
        
        # Remove <think> tags and content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON array or object
        json_match = re.search(r'(\[.*\]|\{.*\})', response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # If no JSON found, return empty array
        return "[]"

    async def _manager_score_fact_extraction(self, result: List[ExtractedFact]) -> float:
        """Manager LLM scores fact extraction quality with fallback"""
        if not result:
            return 0.0
        
        try:
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_fact_extraction(result, self.document_content, config)
            self._update_progress(f"    🎯 Manager evaluated fact extraction: {score:.2f}")
            return score
        except Exception as e:
            self._update_progress(f"    ⚠️ Manager scoring failed, using fallback: {str(e)}")
            return self._fallback_score_fact_extraction(result)

    async def _manager_score_concept_extraction(self, result: List[ExtractedConcept]) -> float:
        """Manager LLM scores concept extraction quality with fallback"""
        if not result:
            return 0.0
        
        try:
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_concept_extraction(result, self.document_content, config)
            self._update_progress(f"    🎯 Manager evaluated concept extraction: {score:.2f}")
            return score
        except Exception as e:
            self._update_progress(f"    ⚠️ Manager scoring failed, using fallback: {str(e)}")
            return self._fallback_score_concept_extraction(result)

    async def _manager_score_qa_generation(self, result: List[Dict[str, str]]) -> float:
        """Manager LLM scores Q&A generation quality with fallback"""
        if not result:
            return 0.0
        
        try:
            # Get concepts for context
            concepts = await self._robust_extract_concepts(self.document_path, {"data_generation_model": self.manager_scorer.manager_model_spec})
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_analytical_qa(result, concepts[:2], config)
            self._update_progress(f"    🎯 Manager evaluated Q&A generation: {score:.2f}")
            return score
        except Exception as e:
            self._update_progress(f"    ⚠️ Manager scoring failed, using fallback: {str(e)}")
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

    async def run_evaluation(self, selected_models: Optional[List[str]] = None):
        """Run the enhanced LLM evaluation with dynamic model discovery"""
        self._update_progress("🚀 Starting Enhanced LLM Evaluation Shoot-out...")
        
        # Discover available models
        discovered_models = await self.discover_ollama_models()
        if not discovered_models:
            self._update_progress("❌ No models found. Aborting evaluation.")
            return
        
        # Filter to text generation models only
        text_gen_models = [m for m in discovered_models if m["type"] == "text_generation"]
        
        if selected_models:
            # Filter to only selected models
            text_gen_models = [m for m in text_gen_models if m["name"] in selected_models]
        
        self._update_progress(f"📊 Evaluating {len(text_gen_models)} text generation models")
        
        total_models = len(text_gen_models)
        for i, model_info in enumerate(text_gen_models):
            model_name = model_info["name"]
            model_spec = f"ollama:{model_name}"
            
            self.current_model = model_name
            self._update_progress(f"\n🤖 [{i+1}/{total_models}] Evaluating Model: {model_spec}")
            
            config = {"data_generation_model": model_spec}
            
            total_tasks = len(self.evaluation_tasks)
            for j, (task_name, task_details) in enumerate(self.evaluation_tasks.items()):
                self.current_task = task_name
                task_progress = ((i * total_tasks + j) / (total_models * total_tasks)) * 100
                
                self._update_progress(f"  📋 [{j+1}/{total_tasks}] Running task: {task_details['description']}", task_progress)
                
                try:
                    # Run the task
                    result = await task_details["function"](self.document_path, config)
                    
                    # Score the result
                    scoring_function = task_details["scoring_function"]
                    if asyncio.iscoroutinefunction(scoring_function):
                        score = await scoring_function(result)
                    else:
                        score = scoring_function(result)
                    
                    self._update_progress(f"    ✅ Task '{task_name}' Score: {score:.2f}")
                    self.llm_manager.update_model_performance(model_spec, task_name, score)
                    
                except Exception as e:
                    self._update_progress(f"    ❌ Task '{task_name}' failed: {str(e)}")
                    self.llm_manager.update_model_performance(model_spec, task_name, 0.0)
        
        self.current_model = None
        self.current_task = None
        self._update_progress("\n🏆 Evaluation Complete!")
        self._update_progress("💾 Scores saved to backend/llm_performance_scores.json")
        self._update_progress("🎯 The Manager Agent will now use these scores for dynamic LLM selection.")

async def main():
    parser = argparse.ArgumentParser(description="Run enhanced LLM evaluation with dynamic model discovery.")
    parser.add_argument("document_path", type=str, help="Path to the document to use for evaluation.")
    parser.add_argument("--ollama-url", type=str, default="http://host.docker.internal:11434", help="Ollama server URL")
    parser.add_argument("--manager-model", type=str, default="ollama:llama3.3:latest", help="Manager model for scoring")
    parser.add_argument("--models", type=str, nargs="*", help="Specific models to evaluate (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.document_path):
        print(f"❌ Error: Document not found at {args.document_path}")
        return

    evaluator = DynamicLLMEvaluator(args.document_path, args.manager_model, args.ollama_url)
    await evaluator.run_evaluation(args.models)

if __name__ == "__main__":
    asyncio.run(main())
