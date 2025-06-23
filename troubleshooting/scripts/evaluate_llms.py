import asyncio
import os
import sys
import argparse
import json
from typing import List, Dict, Any

# Add project root to path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.llm_manager import LLMManager
from backend.improved_alpaca_generator import ImprovedAlpacaGenerator, ExtractedFact, ExtractedConcept
from backend.rag_system import RAGSystem
from backend.json_parser_fix import RobustJSONParser
from backend.manager_scoring_system import ManagerScoringSystem

class LLMEvaluator:
    """A class to evaluate the performance of LLMs on specific tasks."""

    def __init__(self, document_path: str, manager_model: str = "ollama:llama3.3:latest"):
        self.document_path = document_path
        self.llm_manager = LLMManager()
        # A mock RAG system is sufficient for evaluation, as we are focused on generation quality
        self.rag_system = RAGSystem(embedding_model="ollama:bge-m3:latest", reranking_model="ollama:bge-m3:latest")
        self.generator = ImprovedAlpacaGenerator(self.llm_manager, self.rag_system)
        self.manager_scorer = ManagerScoringSystem(self.llm_manager, manager_model)
        self.evaluation_tasks = self._create_evaluation_tasks()
        self.document_content = self._load_document_content()
    
    def _load_document_content(self) -> str:
        """Load document content for manager scoring"""
        try:
            return self.generator._read_document_content(self.document_path)
        except Exception as e:
            print(f"Warning: Could not load document content: {e}")
            return ""

    def _create_evaluation_tasks(self) -> Dict[str, Any]:
        """Create a standardized set of evaluation tasks."""
        return {
            "fact_extraction": {
                "description": "Extract 5 specific, verifiable facts from the document.",
                "function": self.generator._extract_structured_facts,
                "scoring_function": self._manager_score_fact_extraction,
            },
            "concept_extraction": {
                "description": "Extract 3 key concepts from the document.",
                "function": self.generator._extract_structured_concepts,
                "scoring_function": self._manager_score_concept_extraction,
            },
            "analytical_qa": {
                "description": "Generate one analytical Q&A pair based to two concepts.",
                "function": self._generate_analytical_qa,
                "scoring_function": self._manager_score_qa_generation,
            },
        }

    async def _generate_analytical_qa(self, doc_path: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Helper to generate a single analytical Q&A pair for evaluation."""
        concepts = await self.generator._extract_structured_concepts(doc_path, config)
        
        # Filter concepts with valid confidence values (fix the '>=' comparison error)
        valid_concepts = []
        for concept in concepts:
            try:
                # Ensure confidence is a float for comparison
                if isinstance(concept.confidence, str):
                    # Convert string confidence to float
                    confidence_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
                    concept.confidence = confidence_map.get(concept.confidence.lower(), 0.5)
                elif not isinstance(concept.confidence, (int, float)):
                    concept.confidence = 0.5
                
                valid_concepts.append(concept)
            except Exception as e:
                print(f"Warning: Skipping concept due to confidence error: {e}")
                continue
        
        if len(valid_concepts) >= 2:
            return await self.generator._generate_analytical_qa_pairs(valid_concepts[:2], config)
        return []

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
            return self._score_fact_extraction(result)

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
            return self._score_concept_extraction(result)

    async def _manager_score_qa_generation(self, result: List[Dict[str, str]]) -> float:
        """Manager LLM scores Q&A generation quality"""
        if not result:
            return 0.0
        
        try:
            # Get concepts used for context
            concepts = await self.generator._extract_structured_concepts(self.document_path, {"data_generation_model": self.manager_scorer.manager_model_spec})
            config = {"data_generation_model": self.manager_scorer.manager_model_spec}
            score = await self.manager_scorer.score_analytical_qa(result, concepts[:2], config)
            print(f"    - Manager evaluated Q&A generation: {score:.2f}")
            return score
        except Exception as e:
            print(f"    - Manager scoring failed, using fallback: {e}")
            return self._score_qa_generation(result)

    def _score_fact_extraction(self, result: List[ExtractedFact]) -> float:
        """Score the quality of fact extraction with granular metrics."""
        if not result:
            return 0.0
        
        total_score = 0.0
        max_possible_score = 0.0
        
        for fact in result:
            fact_score = 0.0
            fact_max = 5.0  # Maximum possible score per fact
            
            # Content quality (0-2 points)
            if fact.content:
                content_len = len(fact.content.strip())
                if content_len >= 20:  # Substantial content
                    fact_score += 2.0
                elif content_len >= 10:  # Minimal content
                    fact_score += 1.0
                # else: 0 points for very short content
            
            # Context quality (0-1 points)
            if fact.context and len(fact.context.strip()) >= 10:
                fact_score += 1.0
            
            # Fact type specificity (0-1 points)
            if fact.fact_type and fact.fact_type != 'general':
                fact_score += 1.0
            elif fact.fact_type == 'general':
                fact_score += 0.5
            
            # Confidence weighting (0-1 points)
            try:
                confidence = float(fact.confidence) if isinstance(fact.confidence, (int, float)) else 0.5
                fact_score += confidence
            except (ValueError, TypeError):
                fact_score += 0.5  # Default confidence
            
            total_score += fact_score
            max_possible_score += fact_max
        
        # Normalize to 0-1 range
        final_score = total_score / max_possible_score if max_possible_score > 0 else 0.0
        
        # Quantity bonus/penalty (adjust for expected vs actual count)
        expected_facts = 5  # As per task description
        quantity_ratio = min(len(result) / expected_facts, 1.0)  # Cap at 1.0
        
        # Final score combines quality and quantity
        return final_score * 0.8 + quantity_ratio * 0.2

    def _score_concept_extraction(self, result: List[ExtractedConcept]) -> float:
        """Score the quality of concept extraction with granular metrics."""
        if not result:
            return 0.0
        
        total_score = 0.0
        max_possible_score = 0.0
        
        for concept in result:
            concept_score = 0.0
            concept_max = 6.0  # Maximum possible score per concept
            
            # Name quality (0-1 points)
            if concept.name and len(concept.name.strip()) >= 3:
                concept_score += 1.0
            
            # Definition quality (0-2 points)
            if concept.definition:
                def_len = len(concept.definition.strip())
                if def_len >= 50:  # Comprehensive definition
                    concept_score += 2.0
                elif def_len >= 20:  # Basic definition
                    concept_score += 1.0
                elif def_len >= 5:  # Minimal definition
                    concept_score += 0.5
            
            # Examples quality (0-1 points)
            if concept.examples and len(concept.examples) > 0:
                valid_examples = [ex for ex in concept.examples if ex and len(ex.strip()) >= 3]
                if len(valid_examples) >= 2:
                    concept_score += 1.0
                elif len(valid_examples) >= 1:
                    concept_score += 0.5
            
            # Relationships quality (0-1 points)
            if concept.relationships and len(concept.relationships) > 0:
                valid_relationships = [rel for rel in concept.relationships if rel and len(rel.strip()) >= 3]
                if len(valid_relationships) >= 1:
                    concept_score += 1.0
            
            # Confidence weighting (0-1 points)
            try:
                confidence = float(concept.confidence) if isinstance(concept.confidence, (int, float)) else 0.5
                concept_score += confidence
            except (ValueError, TypeError):
                concept_score += 0.5  # Default confidence
            
            total_score += concept_score
            max_possible_score += concept_max
        
        # Normalize to 0-1 range
        final_score = total_score / max_possible_score if max_possible_score > 0 else 0.0
        
        # Quantity bonus/penalty (adjust for expected vs actual count)
        expected_concepts = 3  # As per task description
        quantity_ratio = min(len(result) / expected_concepts, 1.0)  # Cap at 1.0
        
        # Final score combines quality and quantity
        return final_score * 0.8 + quantity_ratio * 0.2

    def _score_qa_generation(self, result: List[Dict[str, str]]) -> float:
        """Score the quality of Q&A generation with granular metrics."""
        if not result:
            return 0.0
        
        total_score = 0.0
        max_possible_score = 0.0
        
        for qa_pair in result:
            pair_score = 0.0
            pair_max = 4.0  # Maximum possible score per Q&A pair
            
            instruction = qa_pair.get("instruction", "")
            output = qa_pair.get("output", "")
            
            # Instruction quality (0-1 points)
            if self.generator._is_valid_instruction(instruction):
                pair_score += 1.0
            
            # Output quality (0-1 points)
            if self.generator._is_valid_output(output):
                pair_score += 1.0
            
            # Pair coherence (0-1 points)
            if self.generator._is_high_quality_pair(instruction, output):
                pair_score += 1.0
            
            # Content depth (0-1 points)
            if instruction and output:
                # Check for analytical depth
                analytical_words = ['compare', 'contrast', 'analyze', 'relationship', 'difference', 'similarity']
                if any(word in instruction.lower() for word in analytical_words):
                    pair_score += 0.5
                
                # Check output comprehensiveness
                if len(output.split()) >= 30:  # Substantial answer
                    pair_score += 0.5
            
            total_score += pair_score
            max_possible_score += pair_max
        
        # Normalize to 0-1 range
        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    async def run_evaluation(self):
        """Run the LLM evaluation 'shoot-out'."""
        print("Starting LLM Evaluation Shoot-out...")
        
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
                    # Handle async scoring functions
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

        print("\n--- Evaluation Complete ---")
        print("Scores have been saved to backend/llm_performance_scores.json")
        print("The Manager Agent will now use these scores for dynamic LLM selection.")

async def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation 'shoot-out'.")
    parser.add_argument("document_path", type=str, help="Path to the document to use for evaluation.")
    args = parser.parse_args()

    if not os.path.exists(args.document_path):
        print(f"Error: Document not found at {args.document_path}")
        return

    evaluator = LLMEvaluator(args.document_path)
    await evaluator.run_evaluation()

if __name__ == "__main__":
    asyncio.run(main())
