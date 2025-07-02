"""
Manager-based scoring system for LLM evaluation
The Manager LLM evaluates worker LLM outputs using consistent, objective criteria
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from backend.llm_manager import LLMManager
from backend.enhanced_alpaca_generator import ExtractedFact, ExtractedConcept

logger = logging.getLogger(__name__)

class ManagerScoringSystem:
    """Manager LLM-based scoring system for objective evaluation"""
    
    def __init__(self, llm_manager: LLMManager, manager_model_spec: str = "ollama:llama3.3:latest"):
        self.llm_manager = llm_manager
        self.manager_model_spec = manager_model_spec
        self.scoring_criteria = self._define_scoring_criteria()
    
    def _define_scoring_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Define objective scoring criteria for each task type"""
        return {
            "fact_extraction": {
                "max_score": 1.0,
                "criteria": {
                    "accuracy": 0.3,      # Is the fact correct and verifiable?
                    "specificity": 0.25,  # Is the fact specific rather than vague?
                    "completeness": 0.2,  # Does it include necessary context?
                    "relevance": 0.15,    # Is it relevant to the document?
                    "clarity": 0.1        # Is it clearly stated?
                }
            },
            "concept_extraction": {
                "max_score": 1.0,
                "criteria": {
                    "definition_quality": 0.35,  # Clear, accurate definition
                    "examples_relevance": 0.25,  # Appropriate examples
                    "relationships": 0.2,        # Valid concept relationships
                    "domain_accuracy": 0.15,     # Correct domain classification
                    "completeness": 0.05         # All required fields present
                }
            },
            "analytical_qa": {
                "max_score": 1.0,
                "criteria": {
                    "question_quality": 0.25,    # Well-formed, analytical question
                    "answer_accuracy": 0.35,     # Correct and comprehensive answer
                    "analytical_depth": 0.25,    # Shows reasoning and analysis
                    "coherence": 0.15            # Question-answer alignment
                }
            }
        }
    
    async def score_fact_extraction(self, facts: List[ExtractedFact], original_document: str, config: Dict[str, Any]) -> float:
        """Manager LLM scores fact extraction quality"""
        if not facts:
            return 0.0
        
        # Prepare facts for evaluation
        facts_text = "\n".join([
            f"Fact {i+1}: {fact.content}\nContext: {fact.context}\nType: {fact.fact_type}\nConfidence: {fact.confidence}"
            for i, fact in enumerate(facts)
        ])
        
        scoring_prompt = f"""You are an expert evaluator assessing the quality of fact extraction from a document.

SCORING CRITERIA (Total: 1.0 points):
- Accuracy (0.3): Are the facts correct and verifiable from the document?
- Specificity (0.25): Are facts specific rather than vague generalizations?
- Completeness (0.2): Do facts include necessary context and details?
- Relevance (0.15): Are facts relevant to the document's main content?
- Clarity (0.1): Are facts clearly and precisely stated?

DOCUMENT EXCERPT:
{original_document[:2000]}...

EXTRACTED FACTS:
{facts_text}

EVALUATION INSTRUCTIONS:
1. Check each fact against the document for accuracy
2. Assess specificity - prefer "The model achieved 95.2% accuracy" over "The model performed well"
3. Evaluate completeness - facts should have sufficient context
4. Consider relevance - facts should relate to key document content
5. Rate clarity - facts should be unambiguous

Provide your assessment in this format:
ACCURACY: [0.0-0.3] - Brief justification
SPECIFICITY: [0.0-0.25] - Brief justification  
COMPLETENESS: [0.0-0.2] - Brief justification
RELEVANCE: [0.0-0.15] - Brief justification
CLARITY: [0.0-0.1] - Brief justification
TOTAL SCORE: [sum of above scores]

Be objective and consistent. Focus on quality over quantity."""

        try:
            response = await self.llm_manager.generate_response(
                self.manager_model_spec, scoring_prompt, config
            )
            return self._parse_score_from_response(response, "fact_extraction")
        except Exception as e:
            logger.error(f"Manager scoring failed for fact extraction: {str(e)}")
            return 0.0
    
    async def score_concept_extraction(self, concepts: List[ExtractedConcept], original_document: str, config: Dict[str, Any]) -> float:
        """Manager LLM scores concept extraction quality"""
        if not concepts:
            return 0.0
        
        # Prepare concepts for evaluation
        concepts_text = "\n".join([
            f"Concept {i+1}: {concept.name}\nDefinition: {concept.definition}\nExamples: {', '.join(concept.examples)}\nRelationships: {', '.join(concept.relationships)}\nDomain: {concept.domain}\nConfidence: {concept.confidence}"
            for i, concept in enumerate(concepts)
        ])
        
        scoring_prompt = f"""You are an expert evaluator assessing the quality of concept extraction from a document.

SCORING CRITERIA (Total: 1.0 points):
- Definition Quality (0.35): Are definitions clear, accurate, and comprehensive?
- Examples Relevance (0.25): Are examples appropriate and illustrative?
- Relationships (0.2): Are concept relationships valid and meaningful?
- Domain Accuracy (0.15): Is domain classification correct?
- Completeness (0.05): Are all required fields properly filled?

DOCUMENT EXCERPT:
{original_document[:2000]}...

EXTRACTED CONCEPTS:
{concepts_text}

EVALUATION INSTRUCTIONS:
1. Assess definition quality - should be clear, accurate, and comprehensive
2. Evaluate examples - should be relevant and help understand the concept
3. Check relationships - should show valid connections between concepts
4. Verify domain classification - should match the concept's field
5. Ensure completeness - all fields should be meaningfully filled

Provide your assessment in this format:
DEFINITION_QUALITY: [0.0-0.35] - Brief justification
EXAMPLES_RELEVANCE: [0.0-0.25] - Brief justification
RELATIONSHIPS: [0.0-0.2] - Brief justification
DOMAIN_ACCURACY: [0.0-0.15] - Brief justification
COMPLETENESS: [0.0-0.05] - Brief justification
TOTAL SCORE: [sum of above scores]

Be objective and consistent. Focus on conceptual accuracy and clarity."""

        try:
            response = await self.llm_manager.generate_response(
                self.manager_model_spec, scoring_prompt, config
            )
            return self._parse_score_from_response(response, "concept_extraction")
        except Exception as e:
            logger.error(f"Manager scoring failed for concept extraction: {str(e)}")
            return 0.0
    
    async def score_analytical_qa(self, qa_pairs: List[Dict[str, str]], concepts_used: List[ExtractedConcept], config: Dict[str, Any]) -> float:
        """Manager LLM scores analytical Q&A quality"""
        if not qa_pairs:
            return 0.0
        
        # Prepare Q&A for evaluation
        qa_text = "\n".join([
            f"Q&A {i+1}:\nQuestion: {qa['instruction']}\nAnswer: {qa['output']}\n"
            for i, qa in enumerate(qa_pairs)
        ])
        
        concepts_context = "\n".join([
            f"Concept: {concept.name} - {concept.definition}"
            for concept in concepts_used
        ])
        
        scoring_prompt = f"""You are an expert evaluator assessing the quality of analytical Q&A pairs.

SCORING CRITERIA (Total: 1.0 points):
- Question Quality (0.25): Is the question well-formed, clear, and analytical?
- Answer Accuracy (0.35): Is the answer correct, comprehensive, and well-reasoned?
- Analytical Depth (0.25): Does it show reasoning, comparison, or analysis?
- Coherence (0.15): Do the question and answer align properly?

CONCEPTS USED:
{concepts_context}

GENERATED Q&A PAIRS:
{qa_text}

EVALUATION INSTRUCTIONS:
1. Assess question quality - should be clear, specific, and require analysis
2. Evaluate answer accuracy - should be correct and comprehensive
3. Check analytical depth - should show reasoning, not just facts
4. Verify coherence - answer should directly address the question

Provide your assessment in this format:
QUESTION_QUALITY: [0.0-0.25] - Brief justification
ANSWER_ACCURACY: [0.0-0.35] - Brief justification
ANALYTICAL_DEPTH: [0.0-0.25] - Brief justification
COHERENCE: [0.0-0.15] - Brief justification
TOTAL SCORE: [sum of above scores]

Be objective and consistent. Focus on analytical rigor and accuracy."""

        try:
            response = await self.llm_manager.generate_response(
                self.manager_model_spec, scoring_prompt, config
            )
            return self._parse_score_from_response(response, "analytical_qa")
        except Exception as e:
            logger.error(f"Manager scoring failed for analytical Q&A: {str(e)}")
            return 0.0
    
    def _parse_score_from_response(self, response: str, task_type: str) -> float:
        """Parse the total score from manager's evaluation response"""
        try:
            # Look for TOTAL SCORE line
            import re
            total_match = re.search(r'TOTAL SCORE:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            if total_match:
                score = float(total_match.group(1))
                max_score = self.scoring_criteria[task_type]["max_score"]
                return min(score, max_score)  # Cap at max score
            
            # Fallback: try to extract individual scores and sum them
            criteria = self.scoring_criteria[task_type]["criteria"]
            total_score = 0.0
            
            for criterion, max_points in criteria.items():
                criterion_upper = criterion.upper().replace("_", "_")
                pattern = f'{criterion_upper}:\\s*([0-9]*\\.?[0-9]+)'
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    total_score += min(score, max_points)
            
            return total_score
            
        except Exception as e:
            logger.error(f"Failed to parse score from manager response: {str(e)}")
            logger.debug(f"Manager response: {response}")
            return 0.0
    
    async def evaluate_model_consistency(self, model_spec: str, task_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model consistency across multiple runs"""
        consistency_prompt = f"""You are evaluating the consistency of an LLM's performance across different tasks.

MODEL: {model_spec}
TASK RESULTS:
{task_results}

CONSISTENCY CRITERIA:
- Output Format Consistency: Does the model consistently follow JSON format requirements?
- Quality Stability: Are the outputs consistently high/low quality or highly variable?
- Response Pattern: Does the model show consistent response patterns?

Rate each criterion from 0.0 to 1.0:
FORMAT_CONSISTENCY: [0.0-1.0] - Brief justification
QUALITY_STABILITY: [0.0-1.0] - Brief justification  
RESPONSE_PATTERN: [0.0-1.0] - Brief justification

Provide objective assessment based on the evidence."""

        try:
            response = await self.llm_manager.generate_response(
                self.manager_model_spec, consistency_prompt, config
            )
            
            # Parse consistency scores
            import re
            consistency_scores = {}
            
            patterns = {
                'format_consistency': r'FORMAT_CONSISTENCY:\s*([0-9]*\.?[0-9]+)',
                'quality_stability': r'QUALITY_STABILITY:\s*([0-9]*\.?[0-9]+)',
                'response_pattern': r'RESPONSE_PATTERN:\s*([0-9]*\.?[0-9]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    consistency_scores[key] = float(match.group(1))
                else:
                    consistency_scores[key] = 0.5  # Default neutral score
            
            return consistency_scores
            
        except Exception as e:
            logger.error(f"Consistency evaluation failed: {str(e)}")
            return {'format_consistency': 0.5, 'quality_stability': 0.5, 'response_pattern': 0.5}
