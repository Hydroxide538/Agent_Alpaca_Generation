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

class LLMEvaluator:
    """A class to evaluate the performance of LLMs on specific tasks."""

    def __init__(self, document_path: str):
        self.document_path = document_path
        self.llm_manager = LLMManager()
        # A mock RAG system is sufficient for evaluation, as we are focused on generation quality
        self.rag_system = RAGSystem(embedding_model="ollama:bge-m3:latest", reranking_model="ollama:bge-m3:latest")
        self.generator = ImprovedAlpacaGenerator(self.llm_manager, self.rag_system)
        self.evaluation_tasks = self._create_evaluation_tasks()

    def _create_evaluation_tasks(self) -> Dict[str, Any]:
        """Create a standardized set of evaluation tasks."""
        return {
            "fact_extraction": {
                "description": "Extract 5 specific, verifiable facts from the document.",
                "function": self.generator._extract_structured_facts,
                "scoring_function": self._score_fact_extraction,
            },
            "concept_extraction": {
                "description": "Extract 3 key concepts from the document.",
                "function": self.generator._extract_structured_concepts,
                "scoring_function": self._score_concept_extraction,
            },
            "analytical_qa": {
                "description": "Generate one analytical Q&A pair based on two concepts.",
                "function": self._generate_analytical_qa,
                "scoring_function": self._score_qa_generation,
            },
        }

    async def _generate_analytical_qa(self, doc_path: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Helper to generate a single analytical Q&A pair for evaluation."""
        concepts = await self.generator._extract_structured_concepts(doc_path, config)
        if len(concepts) >= 2:
            return await self.generator._generate_analytical_qa_pairs(concepts[:2], config)
        return []

    def _score_fact_extraction(self, result: List[ExtractedFact]) -> float:
        """Score the quality of fact extraction."""
        if not result:
            return 0.0
        
        score = 0.0
        for fact in result:
            if fact.content and fact.context and fact.fact_type:
                score += 1.0
            if fact.confidence > 0.7:
                score += 0.5
        return score / (len(result) * 1.5) if result else 0.0

    def _score_concept_extraction(self, result: List[ExtractedConcept]) -> float:
        """Score the quality of concept extraction."""
        if not result:
            return 0.0
        
        score = 0.0
        for concept in result:
            if concept.name and concept.definition:
                score += 1.0
            if concept.examples or concept.relationships:
                score += 0.5
        return score / (len(result) * 1.5) if result else 0.0

    def _score_qa_generation(self, result: List[Dict[str, str]]) -> float:
        """Score the quality of Q&A generation."""
        if not result:
            return 0.0
        
        pair = result[0]
        score = 0.0
        if self.generator._is_valid_instruction(pair.get("instruction", "")):
            score += 1.0
        if self.generator._is_valid_output(pair.get("output", "")):
            score += 1.0
        if self.generator._is_high_quality_pair(pair.get("instruction", ""), pair.get("output", "")):
            score += 1.0
        return score / 3.0

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
                    score = task_details["scoring_function"](result)
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
