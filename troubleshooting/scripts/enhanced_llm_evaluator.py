"""
Enhanced LLM Evaluation System
Addresses thinking model issues, implements batch processing, and includes manager assistance
"""

import asyncio
import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.llm_manager import LLMManager
from backend.improved_alpaca_generator import ImprovedAlpacaGenerator, ExtractedFact, ExtractedConcept
from backend.rag_system import RAGSystem
from backend.json_parser_fix import RobustJSONParser
from backend.manager_scoring_system import ManagerScoringSystem

logger = logging.getLogger(__name__)

class EnhancedLLMEvaluator:
    """Enhanced LLM evaluator with thinking model support and batch processing"""

    def __init__(self, document_path: str, manager_model: str = "ollama:llama3.3:latest"):
        self.document_path = document_path
        self.llm_manager = LLMManager()
        self.rag_system = RAGSystem(embedding_model="ollama:bge-m3:latest", reranking_model="ollama:bge-m3:latest")
        self.generator = ImprovedAlpacaGenerator(self.llm_manager, self.rag_system)
        self.manager_scorer = ManagerScoringSystem(self.llm_manager, manager_model)
        self.evaluation_tasks = self._create_evaluation_tasks()
        self.document_content = self._load_document_content()
        
        # Enhanced configuration for thinking models
        self.thinking_models = {
            "ollama:qwen3:32b", "ollama:reflection:latest", "ollama:phi4-reasoning:latest"
        }
        self.max_retries = 3
        self.thinking_delay = 2  # seconds to let thinking models process
        
        # Batch processing storage
        self.batch_results = {}
        self.manager_assistance_log = []

    def _load_document_content(self) -> str:
        """Load document content for manager scoring"""
        try:
            return self.generator._read_document_content(self.document_path)
        except Exception as e:
            print(f"Warning: Could not load document content: {e}")
            return ""

    def _create_evaluation_tasks(self) -> Dict[str, Any]:
        """Create evaluation tasks with enhanced prompting for thinking models"""
        return {
            "fact_extraction": {
                "description": "Extract 5 specific, verifiable facts from the document.",
                "function": self._enhanced_extract_facts,
                "scoring_function": self._manager_score_fact_extraction,
                "max_score": 1.0
            },
            "concept_extraction": {
                "description": "Extract 3 key concepts from the document.",
                "function": self._enhanced_extract_concepts,
                "scoring_function": self._manager_score_concept_extraction,
                "max_score": 1.0
            },
            "analytical_qa": {
                "description": "Generate one analytical Q&A pair based on two concepts.",
                "function": self._enhanced_generate_analytical_qa,
                "scoring_function": self._manager_score_qa_generation,
                "max_score": 1.0
            },
        }

    async def _enhanced_extract_facts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedFact]:
        """Enhanced fact extraction with thinking model support"""
        model_spec = config.get("data_generation_model", "")
        
        # Check if this is a thinking model
        is_thinking_model = any(thinking_model in model_spec for thinking_model in self.thinking_models)
        
        if is_thinking_model:
            return await self._extract_facts_thinking_model(doc_path, config)
        else:
            return await self.generator._extract_structured_facts(doc_path, config)

    async def _enhanced_extract_concepts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedConcept]:
        """Enhanced concept extraction with thinking model support"""
        model_spec = config.get("data_generation_model", "")
        
        # Check if this is a thinking model
        is_thinking_model = any(thinking_model in model_spec for thinking_model in self.thinking_models)
        
        if is_thinking_model:
            return await self._extract_concepts_thinking_model(doc_path, config)
        else:
            return await self.generator._extract_structured_concepts(doc_path, config)

    async def _enhanced_generate_analytical_qa(self, doc_path: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Enhanced analytical Q&A generation with thinking model support"""
        model_spec = config.get("data_generation_model", "")
        
        # First extract concepts
        concepts = await self._enhanced_extract_concepts(doc_path, config)
        
        # Filter valid concepts
        valid_concepts = []
        for concept in concepts:
            try:
                if isinstance(concept.confidence, str):
                    confidence_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
                    concept.confidence = confidence_map.get(concept.confidence.lower(), 0.5)
                elif not isinstance(concept.confidence, (int, float)):
                    concept.confidence = 0.5
                valid_concepts.append(concept)
            except Exception as e:
                print(f"Warning: Skipping concept due to confidence error: {e}")
                continue
        
        if len(valid_concepts) >= 2:
            # Check if this is a thinking model
            is_thinking_model = any(thinking_model in model_spec for thinking_model in self.thinking_models)
            
            if is_thinking_model:
                return await self._generate_qa_thinking_model(valid_concepts[:2], config)
            else:
                return await self.generator._generate_analytical_qa_pairs(valid_concepts[:2], config)
        
        return []

    async def _extract_facts_thinking_model(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedFact]:
        """Extract facts from thinking models with special handling"""
        content = self.generator._read_document_content(doc_path)
        if not content:
            return []

        chunks = self.generator._split_content_intelligently(content)
        facts = []

        for i, chunk in enumerate(chunks[:3]):  # Limit chunks for thinking models
            # Enhanced prompt for thinking models
            fact_extraction_prompt = f"""You are a fact extraction system. Take your time to think through this task carefully.

TASK: Extract 3-5 specific, verifiable facts from the document section below.

THINKING PROCESS:
1. Read through the document section carefully
2. Identify factual statements that can be verified
3. Ensure each fact is specific and not vague
4. Provide appropriate context for each fact

CRITICAL OUTPUT REQUIREMENTS:
- You MUST respond with ONLY a valid JSON array
- Do NOT include any thinking process in your final response
- Do NOT include explanations, comments, or markdown
- Start your response with [ and end with ]

Document section:
{chunk}

Required JSON format:
[
  {{
    "content": "exact factual statement",
    "context": "surrounding context",
    "fact_type": "numerical",
    "confidence": "high"
  }}
]

Valid fact_type values: numerical, procedural, causal, definitional, categorical, general
Valid confidence values: high, medium, low

Take a moment to think, then provide ONLY the JSON array:"""

            model_spec = config.get("data_generation_model", "")
            
            # Give thinking models more time and multiple attempts
            for attempt in range(self.max_retries):
                try:
                    # Add delay for thinking models
                    if attempt > 0:
                        await asyncio.sleep(self.thinking_delay)
                    
                    response = await self.llm_manager.generate_response(model_spec, fact_extraction_prompt, config)
                    
                    # Enhanced JSON extraction for thinking models
                    chunk_facts_data = self._extract_json_from_thinking_response(response)
                    
                    if chunk_facts_data and RobustJSONParser.validate_extracted_facts(chunk_facts_data):
                        chunk_facts = []
                        for f in chunk_facts_data:
                            confidence = f.get('confidence', 'medium')
                            if isinstance(confidence, str):
                                confidence = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(confidence.lower(), 0.5)
                            
                            fact = ExtractedFact(
                                content=f.get('content', ''),
                                context=f.get('context', ''),
                                confidence=confidence,
                                source_location=f"{doc_path}:chunk_{i}",
                                fact_type=f.get('fact_type', 'general')
                            )
                            chunk_facts.append(fact)
                        
                        facts.extend(chunk_facts)
                        break  # Success, break retry loop
                    else:
                        print(f"Failed to extract or validate JSON facts from LLM response. Response: {response[:500]}")
                        if attempt == self.max_retries - 1:
                            # Last attempt - try manager assistance
                            assisted_response = await self._get_manager_assistance(
                                model_spec, fact_extraction_prompt, response, "fact_extraction"
                            )
                            if assisted_response:
                                chunk_facts_data = self._extract_json_from_thinking_response(assisted_response)
                                if chunk_facts_data and RobustJSONParser.validate_extracted_facts(chunk_facts_data):
                                    chunk_facts = []
                                    for f in chunk_facts_data:
                                        confidence = f.get('confidence', 'medium')
                                        if isinstance(confidence, str):
                                            confidence = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(confidence.lower(), 0.5)
                                        
                                        fact = ExtractedFact(
                                            content=f.get('content', ''),
                                            context=f.get('context', ''),
                                            confidence=confidence,
                                            source_location=f"{doc_path}:chunk_{i}",
                                            fact_type=f.get('fact_type', 'general')
                                        )
                                        chunk_facts.append(fact)
                                    facts.extend(chunk_facts)
                
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for fact extraction: {e}")
                    if attempt == self.max_retries - 1:
                        print(f"All attempts failed for chunk {i}")

        return facts

    async def _extract_concepts_thinking_model(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedConcept]:
        """Extract concepts from thinking models with special handling"""
        content = self.generator._read_document_content(doc_path)
        if not content:
            return []

        chunks = self.generator._split_content_intelligently(content)
        concepts = []

        for i, chunk in enumerate(chunks[:3]):  # Limit chunks for thinking models
            concept_extraction_prompt = f"""You are a concept extraction system. Take your time to think through this task carefully.

TASK: Extract 2-4 key concepts from the document section below.

THINKING PROCESS:
1. Read through the document section carefully
2. Identify the main concepts and ideas
3. Provide clear definitions for each concept
4. Include relevant examples and relationships

CRITICAL OUTPUT REQUIREMENTS:
- You MUST respond with ONLY a valid JSON array
- Do NOT include any thinking process in your final response
- Do NOT include explanations, comments, or markdown
- Start your response with [ and end with ]

Document section:
{chunk}

Required JSON format:
[
  {{
    "name": "concept name",
    "definition": "clear definition",
    "examples": ["example1", "example2"],
    "relationships": ["relationship1"],
    "domain": "field name",
    "confidence": "high"
  }}
]

Valid confidence values: high, medium, low

Take a moment to think, then provide ONLY the JSON array:"""

            model_spec = config.get("data_generation_model", "")
            
            # Give thinking models more time and multiple attempts
            for attempt in range(self.max_retries):
                try:
                    if attempt > 0:
                        await asyncio.sleep(self.thinking_delay)
                    
                    response = await self.llm_manager.generate_response(model_spec, concept_extraction_prompt, config)
                    
                    chunk_concepts_data = self._extract_json_from_thinking_response(response)
                    
                    if chunk_concepts_data and RobustJSONParser.validate_extracted_concepts(chunk_concepts_data):
                        chunk_concepts = []
                        for c in chunk_concepts_data:
                            confidence = c.get('confidence', 'medium')
                            if isinstance(confidence, str):
                                confidence = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(confidence.lower(), 0.5)
                            
                            concept = ExtractedConcept(
                                name=c.get('name', ''),
                                definition=c.get('definition', ''),
                                examples=c.get('examples', []),
                                relationships=c.get('relationships', []),
                                domain=c.get('domain', 'general'),
                                confidence=confidence
                            )
                            chunk_concepts.append(concept)
                        
                        concepts.extend(chunk_concepts)
                        break
                    else:
                        print(f"Failed to extract or validate JSON concepts from LLM response. Response: {response[:500]}")
                        if attempt == self.max_retries - 1:
                            # Last attempt - try manager assistance
                            assisted_response = await self._get_manager_assistance(
                                model_spec, concept_extraction_prompt, response, "concept_extraction"
                            )
                            if assisted_response:
                                chunk_concepts_data = self._extract_json_from_thinking_response(assisted_response)
                                if chunk_concepts_data and RobustJSONParser.validate_extracted_concepts(chunk_concepts_data):
                                    chunk_concepts = []
                                    for c in chunk_concepts_data:
                                        confidence = c.get('confidence', 'medium')
                                        if isinstance(confidence, str):
                                            confidence = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(confidence.lower(), 0.5)
                                        
                                        concept = ExtractedConcept(
                                            name=c.get('name', ''),
                                            definition=c.get('definition', ''),
                                            examples=c.get('examples', []),
                                            relationships=c.get('relationships', []),
                                            domain=c.get('domain', 'general'),
                                            confidence=confidence
                                        )
                                        chunk_concepts.append(concept)
                                    concepts.extend(chunk_concepts)
                
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for concept extraction: {e}")
                    if attempt == self.max_retries - 1:
                        print(f"All attempts failed for chunk {i}")

        return concepts

    async def _generate_qa_thinking_model(self, concepts: List[ExtractedConcept], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate Q&A from thinking models with special handling"""
        if len(concepts) < 2:
            return []

        concept1, concept2 = concepts[0], concepts[1]
        
        qa_prompt = f"""You are a Q&A generation system. Take your time to think through this task carefully.

TASK: Generate ONE analytical Q&A pair that compares or relates these two concepts.

THINKING PROCESS:
1. Understand both concepts thoroughly
2. Identify meaningful relationships or comparisons
3. Create an analytical question that requires reasoning
4. Provide a comprehensive answer

CRITICAL OUTPUT REQUIREMENTS:
- You MUST respond with ONLY a valid JSON array
- Do NOT include any thinking process in your final response
- Do NOT include explanations, comments, or markdown
- Start your response with [ and end with ]

Concept 1: {concept1.name}
Definition: {concept1.definition}
Examples: {', '.join(concept1.examples) if concept1.examples else 'None'}

Concept 2: {concept2.name}
Definition: {concept2.definition}
Examples: {', '.join(concept2.examples) if concept2.examples else 'None'}

Required JSON format:
[
  {{
    "instruction": "analytical question comparing or relating the concepts",
    "input": "",
    "output": "comprehensive answer that shows reasoning and analysis"
  }}
]

Take a moment to think, then provide ONLY the JSON array:"""

        model_spec = config.get("data_generation_model", "")
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.thinking_delay)
                
                response = await self.llm_manager.generate_response(model_spec, qa_prompt, config)
                
                qa_data = self._extract_json_from_thinking_response(response)
                
                if qa_data and isinstance(qa_data, list) and len(qa_data) > 0:
                    qa_pair = qa_data[0]
                    if isinstance(qa_pair, dict) and 'instruction' in qa_pair and 'output' in qa_pair:
                        return [qa_pair]
                
                if attempt == self.max_retries - 1:
                    # Last attempt - try manager assistance
                    assisted_response = await self._get_manager_assistance(
                        model_spec, qa_prompt, response, "analytical_qa"
                    )
                    if assisted_response:
                        qa_data = self._extract_json_from_thinking_response(assisted_response)
                        if qa_data and isinstance(qa_data, list) and len(qa_data) > 0:
                            qa_pair = qa_data[0]
                            if isinstance(qa_pair, dict) and 'instruction' in qa_pair and 'output' in qa_pair:
                                return [qa_pair]
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for Q&A generation: {e}")

        return []

    def _extract_json_from_thinking_response(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """Enhanced JSON extraction specifically for thinking models"""
        if not response:
            return None
        
        # Remove thinking tags and content
        import re
        
        # Remove various thinking patterns
        thinking_patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'Looking at.*?(?=\[|\{)',
            r'Okay.*?(?=\[|\{)',
            r'Let me.*?(?=\[|\{)',
            r'I need to.*?(?=\[|\{)',
            r'First.*?(?=\[|\{)',
        ]
        
        cleaned_response = response
        for pattern in thinking_patterns:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Use the robust JSON parser
        return RobustJSONParser.extract_json_from_response(cleaned_response)

    async def _get_manager_assistance(self, model_spec: str, original_prompt: str, failed_response: str, task_type: str) -> Optional[str]:
        """Get assistance from manager LLM for struggling models"""
        assistance_prompt = f"""You are a Manager LLM helping another model understand a task better.

The model {model_spec} is struggling with a {task_type} task. Here's what happened:

ORIGINAL TASK:
{original_prompt[:1000]}...

MODEL'S RESPONSE:
{failed_response[:500]}...

PROBLEM ANALYSIS:
The model seems to be having trouble with:
1. Following JSON format requirements
2. Providing clean output without thinking process
3. Understanding the specific task requirements

Please provide a SIMPLIFIED and CLEARER version of the original prompt that will help this model succeed. Focus on:
- Very clear JSON format requirements
- Simple, direct instructions
- Examples if helpful
- Emphasis on output format

Provide the improved prompt:"""

        try:
            improved_prompt = await self.llm_manager.generate_response(
                self.manager_scorer.manager_model_spec, assistance_prompt, {}
            )
            
            # Log the assistance
            self.manager_assistance_log.append({
                "timestamp": datetime.now().isoformat(),
                "model": model_spec,
                "task_type": task_type,
                "original_prompt_length": len(original_prompt),
                "failed_response_length": len(failed_response),
                "assistance_provided": True
            })
            
            print(f"    - Manager provided assistance for {model_spec} on {task_type}")
            
            # Try the improved prompt
            config = {"data_generation_model": model_spec}
            return await self.llm_manager.generate_response(model_spec, improved_prompt, config)
            
        except Exception as e:
            print(f"    - Manager assistance failed: {e}")
            self.manager_assistance_log.append({
                "timestamp": datetime.now().isoformat(),
                "model": model_spec,
                "task_type": task_type,
                "assistance_provided": False,
                "error": str(e)
            })
            return None

    async def _manager_score_fact_extraction(self, result: List[ExtractedFact]) -> float:
        """Manager LLM scores fact extraction quality"""
        if not result:
            return 0.0
        
        try:
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_fact_extraction(result, self.document_content, config)
            print(f"    - Manager evaluated fact extraction: {score:.2f}")
            return score
        except Exception as e:
            print(f"    - Manager scoring failed, using fallback: {e}")
            return self._fallback_score_facts(result)

    async def _manager_score_concept_extraction(self, result: List[ExtractedConcept]) -> float:
        """Manager LLM scores concept extraction quality"""
        if not result:
            return 0.0
        
        try:
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_concept_extraction(result, self.document_content, config)
            print(f"    - Manager evaluated concept extraction: {score:.2f}")
            return score
        except Exception as e:
            print(f"    - Manager scoring failed, using fallback: {e}")
            return self._fallback_score_concepts(result)

    async def _manager_score_qa_generation(self, result: List[Dict[str, str]]) -> float:
        """Manager LLM scores Q&A generation quality"""
        if not result:
            return 0.0
        
        try:
            concepts = await self._enhanced_extract_concepts(self.document_path, {"data_generation_model": self.manager_scorer.manager_model_spec})
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_analytical_qa(result, concepts[:2], config)
            print(f"    - Manager evaluated Q&A generation: {score:.2f}")
            return score
        except Exception as e:
            print(f"    - Manager scoring failed, using fallback: {e}")
            return self._fallback_score_qa(result)

    def _fallback_score_facts(self, facts: List[ExtractedFact]) -> float:
        """Fallback scoring for facts when manager fails"""
        if not facts:
            return 0.0
        
        total_score = 0.0
        for fact in facts:
            score = 0.0
            if fact.content and len(fact.content) > 10:
                score += 0.4
            if fact.context and len(fact.context) > 5:
                score += 0.3
            if fact.fact_type and fact.fact_type != 'general':
                score += 0.2
            if fact.confidence > 0.5:
                score += 0.1
            total_score += score
        
        return min(total_score / len(facts), 1.0)

    def _fallback_score_concepts(self, concepts: List[ExtractedConcept]) -> float:
        """Fallback scoring for concepts when manager fails"""
        if not concepts:
            return 0.0
        
        total_score = 0.0
        for concept in concepts:
            score = 0.0
            if concept.name and len(concept.name) > 3:
                score += 0.3
            if concept.definition and len(concept.definition) > 10:
                score += 0.4
            if concept.examples:
                score += 0.2
            if concept.confidence > 0.5:
                score += 0.1
            total_score += score
        
        return min(total_score / len(concepts), 1.0)

    def _fallback_score_qa(self, qa_pairs: List[Dict[str, str]]) -> float:
        """Fallback scoring for Q&A when manager fails"""
        if not qa_pairs:
            return 0.0
        
        total_score = 0.0
        for qa in qa_pairs:
            score = 0.0
            instruction = qa.get('instruction', '')
            output = qa.get('output', '')
            
            if instruction and len(instruction) > 10:
                score += 0.3
            if output and len(output) > 20:
                score += 0.4
            if '?' in instruction:
                score += 0.2
            if len(output.split()) > 10:
                score += 0.1
            total_score += score
        
        return min(total_score / len(qa_pairs), 1.0)

    async def run_batch_evaluation(self) -> Dict[str, Any]:
        """Run batch evaluation - complete all tasks per model, then score all at once"""
        print("Starting Enhanced LLM Evaluation with Batch Processing...")
        
        ollama_models = self.llm_manager.get_available_ollama_models_from_registry()
        if not ollama_models:
            print("No Ollama models found in the registry. Aborting.")
            return {}

        evaluation_results = {}
        
        # Phase 1: Execute all tasks for each model
        print("\n=== PHASE 1: TASK EXECUTION ===")
        for model_info in ollama_models:
            if model_info.get("type") != "text_generation":
                print(f"\n--- Skipping non-text generation model: {model_info['name']} ---")
                continue

            model_spec = f"ollama:{model_info['name']}"
            print(f"\n--- Executing Tasks for Model: {model_spec} ---")
            
            config = {"data_generation_model": model_spec}
            model_results = {}
            
            for task_name, task_details in self.evaluation_tasks.items():
                print(f"  - Executing task: {task_details['description']}")
                start_time = time.time()
                
                try:
                    result = await task_details["function"](self.document_path, config)
                    execution_time = time.time() - start_time
                    
                    model_results[task_name] = {
                        "result": result,
                        "execution_time": execution_time,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"    - Task '{task_name}' completed in {execution_time:.2f}s")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    model_results[task_name] = {
                        "result": None,
                        "execution_time": execution_time,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"    - Task '{task_name}' failed: {e}")
            
            evaluation_results[model_spec] = model_results
            
            # Save intermediate results
            self._save_batch_results(evaluation_results, "intermediate")

        # Phase 2: Manager scoring for all completed tasks
        print("\n=== PHASE 2: MANAGER SCORING ===")
        final_scores = {}
        
        for model_spec, model_results in evaluation_results.items():
            print(f"\n--- Manager Scoring for Model: {model_spec} ---")
            model_scores = {}
            
            for task_name, task_result in model_results.items():
                if task_result["status"] == "completed" and task_result["result"] is not None:
                    print(f"  - Scoring task: {task_name}")
                    
                    try:
                        task_details = self.evaluation_tasks[task_name]
                        scoring_function = task_details["scoring_function"]
                        
                        if asyncio.iscoroutinefunction(scoring_function):
                            score = await scoring_function(task_result["result"])
                        else:
                            score = scoring_function(task_result["result"])
                        
                        model_scores[task_name] = {
                            "score": score,
                            "max_score": task_details["max_score"],
                            "percentage": (score / task_details["max_score"]) * 100 if task_details["max_score"] > 0 else 0
                        }
                        
                        print(f"    - Task '{task_name}' Score: {score:.2f}/{task_details['max_score']:.2f} ({model_scores[task_name]['percentage']:.1f}%)")
                        
                        # Update LLM manager performance tracking
                        self.llm_manager.update_model_performance(model_spec, task_name, score)
                        
                    except Exception as e:
                        print(f"    - Scoring failed for task '{task_name}': {e}")
                        model_scores[task_name] = {
                            "score": 0.0,
                            "max_score": task_details["max_score"],
                            "percentage": 0.0,
                            "error": str(e)
                        }
                        self.llm_manager.update_model_performance(model_spec, task_name, 0.0)
                else:
                    print(f"  - Skipping task '{task_name}' (status: {task_result['status']})")
                    model_scores[task_name] = {
                        "score": 0.0,
                        "max_score": self.evaluation_tasks[task_name]["max_score"],
                        "percentage": 0.0,
                        "status": task_result["status"]
                    }
                    self.llm_manager.update_model_performance(model_spec, task_name, 0.0)
            
            # Calculate overall model score
            total_score = sum(score_info["score"] for score_info in model_scores.values())
            max_total_score = sum(score_info["max_score"] for score_info in model_scores.values())
            overall_percentage = (total_score / max_total_score) * 100 if max_total_score > 0 else 0
            
            final_scores[model_spec] = {
                "individual_scores": model_scores,
                "total_score": total_score,
                "max_total_score": max_total_score,
                "overall_percentage": overall_percentage,
                "execution_results": model_results
            }
            
            print(f"  - Overall Score: {total_score:.2f}/{max_total_score:.2f} ({overall_percentage:.1f}%)")

        # Phase 3: Generate comprehensive report
        print("\n=== PHASE 3: GENERATING REPORT ===")
        report = self._generate_evaluation_report(final_scores)
        
        # Save final results
        self._save_batch_results(final_scores, "final")
        self._save_evaluation_report(report)
        
        print("\n--- Enhanced Evaluation Complete ---")
        print("Results saved to:")
        print("- backend/enhanced_llm_evaluation_results.json")
        print("- backend/enhanced_llm_evaluation_report.json")
        if self.manager_assistance_log:
            print(f"- Manager provided assistance {len(self.manager_assistance_log)} times")
        
        return final_scores

    def _save_batch_results(self, results: Dict[str, Any], result_type: str):
        """Save batch results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backend/enhanced_llm_evaluation_{result_type}_{timestamp}.json"
            
            # Convert any non-serializable objects to dictionaries
            serializable_results = self._make_serializable(results)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"  - {result_type.title()} results saved to {filename}")
            
        except Exception as e:
            print(f"  - Failed to save {result_type} results: {e}")

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Convert dataclass or object to dict
            return self._make_serializable(obj.__dict__)
        else:
            return obj

    def _generate_evaluation_report(self, final_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "document_path": self.document_path,
                "manager_model": self.manager_scorer.manager_model_spec,
                "evaluation_type": "enhanced_batch_processing",
                "thinking_models_detected": list(self.thinking_models),
                "manager_assistance_count": len(self.manager_assistance_log)
            },
            "model_rankings": [],
            "task_performance": {},
            "thinking_model_analysis": {},
            "manager_assistance_summary": self._summarize_manager_assistance(),
            "recommendations": []
        }
        
        # Create model rankings
        model_rankings = []
        for model_spec, scores in final_scores.items():
            model_rankings.append({
                "model": model_spec,
                "overall_score": scores["total_score"],
                "overall_percentage": scores["overall_percentage"],
                "individual_scores": scores["individual_scores"]
            })
        
        # Sort by overall score
        model_rankings.sort(key=lambda x: x["overall_score"], reverse=True)
        report["model_rankings"] = model_rankings
        
        # Analyze task performance across models
        for task_name in self.evaluation_tasks.keys():
            task_scores = []
            for model_spec, scores in final_scores.items():
                if task_name in scores["individual_scores"]:
                    task_scores.append({
                        "model": model_spec,
                        "score": scores["individual_scores"][task_name]["score"],
                        "percentage": scores["individual_scores"][task_name]["percentage"]
                    })
            
            task_scores.sort(key=lambda x: x["score"], reverse=True)
            report["task_performance"][task_name] = {
                "best_model": task_scores[0] if task_scores else None,
                "worst_model": task_scores[-1] if task_scores else None,
                "average_score": sum(s["score"] for s in task_scores) / len(task_scores) if task_scores else 0,
                "all_scores": task_scores
            }
        
        # Analyze thinking models specifically
        for model_spec, scores in final_scores.items():
            is_thinking = any(thinking_model in model_spec for thinking_model in self.thinking_models)
            if is_thinking:
                report["thinking_model_analysis"][model_spec] = {
                    "overall_performance": scores["overall_percentage"],
                    "assistance_received": len([log for log in self.manager_assistance_log if log["model"] == model_spec]),
                    "task_breakdown": scores["individual_scores"]
                }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(final_scores)
        
        return report

    def _summarize_manager_assistance(self) -> Dict[str, Any]:
        """Summarize manager assistance activities"""
        if not self.manager_assistance_log:
            return {"total_assistance": 0, "models_helped": [], "task_breakdown": {}}
        
        models_helped = set()
        task_breakdown = {}
        successful_assistance = 0
        
        for log in self.manager_assistance_log:
            models_helped.add(log["model"])
            task_type = log["task_type"]
            
            if task_type not in task_breakdown:
                task_breakdown[task_type] = {"count": 0, "successful": 0}
            
            task_breakdown[task_type]["count"] += 1
            
            if log.get("assistance_provided", False):
                successful_assistance += 1
                task_breakdown[task_type]["successful"] += 1
        
        return {
            "total_assistance": len(self.manager_assistance_log),
            "successful_assistance": successful_assistance,
            "models_helped": list(models_helped),
            "task_breakdown": task_breakdown,
            "success_rate": (successful_assistance / len(self.manager_assistance_log)) * 100 if self.manager_assistance_log else 0
        }

    def _generate_recommendations(self, final_scores: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Find best overall model
        best_model = max(final_scores.items(), key=lambda x: x[1]["total_score"])
        recommendations.append(f"Best overall model: {best_model[0]} with {best_model[1]['overall_percentage']:.1f}% performance")
        
        # Find best model per task
        for task_name in self.evaluation_tasks.keys():
            task_best = None
            best_score = 0
            
            for model_spec, scores in final_scores.items():
                if task_name in scores["individual_scores"]:
                    score = scores["individual_scores"][task_name]["score"]
                    if score > best_score:
                        best_score = score
                        task_best = model_spec
            
            if task_best:
                recommendations.append(f"Best model for {task_name}: {task_best}")
        
        # Thinking model recommendations
        thinking_models_performance = []
        for model_spec, scores in final_scores.items():
            is_thinking = any(thinking_model in model_spec for thinking_model in self.thinking_models)
            if is_thinking:
                thinking_models_performance.append((model_spec, scores["overall_percentage"]))
        
        if thinking_models_performance:
            thinking_models_performance.sort(key=lambda x: x[1], reverse=True)
            best_thinking = thinking_models_performance[0]
            recommendations.append(f"Best thinking model: {best_thinking[0]} with {best_thinking[1]:.1f}% performance")
            
            if len(thinking_models_performance) > 1:
                avg_thinking_performance = sum(p[1] for p in thinking_models_performance) / len(thinking_models_performance)
                recommendations.append(f"Average thinking model performance: {avg_thinking_performance:.1f}%")
        
        # Manager assistance recommendations
        if self.manager_assistance_log:
            assistance_summary = self._summarize_manager_assistance()
            if assistance_summary["success_rate"] < 50:
                recommendations.append("Consider improving manager assistance prompts - low success rate detected")
            
            most_helped_models = [log["model"] for log in self.manager_assistance_log]
            if most_helped_models:
                from collections import Counter
                most_common = Counter(most_helped_models).most_common(1)[0]
                recommendations.append(f"Model requiring most assistance: {most_common[0]} ({most_common[1]} times)")
        
        return recommendations

    def _save_evaluation_report(self, report: Dict[str, Any]):
        """Save evaluation report to JSON file"""
        try:
            filename = "backend/enhanced_llm_evaluation_report.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  - Evaluation report saved to {filename}")
        except Exception as e:
            print(f"  - Failed to save evaluation report: {e}")

    async def run_legacy_evaluation(self):
        """Run legacy evaluation for compatibility"""
        print("Running legacy evaluation mode...")
        
        ollama_models = self.llm_manager.get_available_ollama_models_from_registry()
        if not ollama_models:
            print("No Ollama models found in the registry. Aborting.")
            return

        for model_info in ollama_models:
            if model_info.get("type") != "text_generation":
                print(f"\n--- Skipping non-text generation model: {model_info['name']} ---")
                continue

            model_spec = f"ollama:{model_info['name']}"
            print(f"\n--- Evaluating Model: {model_spec} ---")
            
            config = {"data_generation_model": model_spec}
            
            for task_name, task_details in self.evaluation_tasks.items():
                print(f"  - Running task: {task_details['description']}")
                try:
                    result = await task_details["function"](self.document_path, config)
                    scoring_function = task_details["scoring_function"]
                    if asyncio.iscoroutinefunction(scoring_function):
                        score = await scoring_function(result)
                    else:
                        score = scoring_function(result)
                    print(f"    - Task '{task_name}' Score: {score:.2f}")
                    self.llm_manager.update_model_performance(model_spec, task_name, score)
                except Exception as e:
                    print(f"    - Task '{task_name}' failed: {e}")
                    self.llm_manager.update_model_performance(model_spec, task_name, 0.0)

        print("\n--- Legacy Evaluation Complete ---")


async def main():
    parser = argparse.ArgumentParser(description="Run enhanced LLM evaluation with thinking model support.")
    parser.add_argument("document_path", type=str, help="Path to the document to use for evaluation.")
    parser.add_argument("--mode", type=str, choices=["batch", "legacy"], default="batch", 
                       help="Evaluation mode: batch (new) or legacy (original)")
    parser.add_argument("--manager-model", type=str, default="ollama:llama3.3:latest",
                       help="Manager model for scoring and assistance")
    args = parser.parse_args()

    if not os.path.exists(args.document_path):
        print(f"Error: Document not found at {args.document_path}")
        return

    evaluator = EnhancedLLMEvaluator(args.document_path, args.manager_model)
    
    if args.mode == "batch":
        await evaluator.run_batch_evaluation()
    else:
        await evaluator.run_legacy_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
