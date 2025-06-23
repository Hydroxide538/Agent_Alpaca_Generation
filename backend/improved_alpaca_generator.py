import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import re
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractedFact:
    """Structured representation of an extracted fact"""
    content: str
    context: str
    confidence: float
    source_location: str
    fact_type: str  # numerical, categorical, procedural, etc.

@dataclass
class ExtractedConcept:
    """Structured representation of an extracted concept"""
    name: str
    definition: str
    examples: List[str]
    relationships: List[str]
    domain: str
    confidence: float

class ImprovedAlpacaGenerator:
    """Generate high-quality training data in Alpaca format from documents"""
    
    def __init__(self, llm_manager, rag_system):
        self.llm_manager = llm_manager
        self.rag_system = rag_system
        self.instruction_templates = self._load_instruction_templates()
        self.quality_filters = self._setup_quality_filters()
    
    def _load_instruction_templates(self) -> Dict[str, List[str]]:
        """Load diverse, natural instruction templates"""
        return {
            "factual_direct": [
                "What is {concept}?",
                "Define {concept}.",
                "Explain what {concept} means.",
                "Describe {concept} in detail.",
                "What are the key characteristics of {concept}?",
                "Provide an overview of {concept}.",
                "What should I know about {concept}?",
                "Give me information about {concept}."
            ],
            "factual_specific": [
                "According to the document, what is {specific_fact}?",
                "The document mentions {topic}. What details are provided?",
                "What specific information is given about {concept}?",
                "What data is presented regarding {topic}?",
                "What findings are reported about {concept}?",
                "What evidence is provided for {claim}?",
                "What statistics or numbers are mentioned for {topic}?"
            ],
            "analytical": [
                "How does {concept1} relate to {concept2}?",
                "What is the relationship between {concept1} and {concept2}?",
                "Compare and contrast {concept1} and {concept2}.",
                "What are the similarities and differences between {concept1} and {concept2}?",
                "How do {concept1} and {concept2} interact?",
                "Analyze the connection between {concept1} and {concept2}.",
                "What can we learn by comparing {concept1} and {concept2}?"
            ],
            "application": [
                "How can {concept} be applied in practice?",
                "What are real-world applications of {concept}?",
                "Give examples of how {concept} is used.",
                "How would you implement {concept}?",
                "What are practical use cases for {concept}?",
                "How can organizations benefit from {concept}?",
                "What are the implications of {concept} for {domain}?"
            ],
            "procedural": [
                "How do you {process}?",
                "What are the steps to {process}?",
                "Walk me through the process of {process}.",
                "What is the procedure for {process}?",
                "How would you go about {process}?",
                "What's involved in {process}?",
                "Explain the methodology for {process}."
            ],
            "evaluative": [
                "What are the advantages and disadvantages of {concept}?",
                "Evaluate the effectiveness of {concept}.",
                "What are the pros and cons of {concept}?",
                "Assess the benefits and limitations of {concept}.",
                "What are the strengths and weaknesses of {concept}?",
                "How effective is {concept} and why?",
                "What are the trade-offs involved with {concept}?"
            ],
            "causal": [
                "What causes {effect}?",
                "Why does {effect} occur?",
                "What factors lead to {effect}?",
                "What are the reasons behind {effect}?",
                "How does {cause} result in {effect}?",
                "What contributes to {effect}?",
                "Explain the causal relationship between {cause} and {effect}."
            ],
            "contextual": [
                "In the context of {domain}, how is {concept} relevant?",
                "What role does {concept} play in {context}?",
                "How does {concept} fit into the broader picture of {domain}?",
                "Why is {concept} important for {context}?",
                "What significance does {concept} have in {domain}?",
                "How does understanding {concept} help with {application}?"
            ]
        }
    
    def _setup_quality_filters(self) -> Dict[str, Any]:
        """Setup quality filtering criteria"""
        return {
            "min_instruction_length": 15,
            "min_output_length": 50,
            "max_output_length": 2000,
            "forbidden_phrases": [
                "I am not a certified professional",
                "Disclaimer:",
                "I cannot provide",
                "I don't have access",
                "<think>",
                "</think>",
                "Based on this fact:",
                "The concept name",
                "Definition:",
                "Create a practical application question"
            ],
            "required_content_ratio": 0.7,  # 70% of output should be actual content
            "max_repetition_ratio": 0.3  # Max 30% repetitive content
        }
    
    async def generate_alpaca_dataset(self, document_paths: List[str], config: Dict[str, Any], websocket_manager=None) -> Dict[str, Any]:
        """Generate high-quality Alpaca format dataset from documents"""
        try:
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Starting improved Alpaca dataset generation..."
                })
            
            # Step 1: Process documents with RAG
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Processing documents and creating embeddings..."
                })
            
            rag_results = await self.rag_system.process_documents(document_paths, websocket_manager)
            
            # Step 2: Extract high-quality facts and concepts
            all_facts = []
            all_concepts = []
            
            for doc_path in document_paths:
                if websocket_manager:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "info",
                        "message": f"Extracting knowledge from: {doc_path}"
                    })
                
                doc_facts = await self._extract_structured_facts(doc_path, config)
                doc_concepts = await self._extract_structured_concepts(doc_path, config)
                
                all_facts.extend(doc_facts)
                all_concepts.extend(doc_concepts)
            
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": f"Extracted {len(all_facts)} facts and {len(all_concepts)} concepts"
                })
            
            # Step 3: Generate diverse, high-quality Q&A pairs
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Generating diverse Q&A pairs..."
                })
            
            alpaca_data = []
            
            # Generate different types of questions with quality control
            alpaca_data.extend(await self._generate_factual_qa_pairs(all_facts, config))
            alpaca_data.extend(await self._generate_conceptual_qa_pairs(all_concepts, config))
            alpaca_data.extend(await self._generate_analytical_qa_pairs(all_concepts, config))
            alpaca_data.extend(await self._generate_application_qa_pairs(all_concepts, config))
            alpaca_data.extend(await self._generate_document_qa_pairs(document_paths, config))
            
            # Step 4: Apply rigorous quality filtering
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Applying quality filters..."
                })
            
            high_quality_data = self._apply_quality_filters(alpaca_data)
            
            # Step 5: Diversify and balance the dataset
            balanced_data = self._balance_dataset(high_quality_data)
            
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"Generated {len(balanced_data)} high-quality Alpaca training examples"
                })
            
            return {
                "alpaca_data": balanced_data,
                "statistics": {
                    "total_examples": len(balanced_data),
                    "documents_processed": len(document_paths),
                    "facts_extracted": len(all_facts),
                    "concepts_extracted": len(all_concepts),
                    "quality_filtered": len(alpaca_data) - len(high_quality_data),
                    "final_dataset_size": len(balanced_data)
                },
                "quality_metrics": self._calculate_quality_metrics(balanced_data),
                "metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": config.get("data_generation_model"),
                    "quality_filters_applied": True,
                    "diversity_balanced": True
                }
            }
            
        except Exception as e:
            logger.error(f"Improved Alpaca dataset generation failed: {str(e)}")
            raise
    
    async def _extract_structured_facts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedFact]:
        """Extract structured, high-quality facts from document"""
        try:
            content = self._read_document_content(doc_path)
            if not content:
                return []
            
            # Split content into chunks for better processing
            chunks = self._split_content_intelligently(content)
            facts = []
            
            for i, chunk in enumerate(chunks[:5]):  # Process first 5 chunks
                fact_extraction_prompt = f"""
                Analyze this document section and extract specific, verifiable facts. Focus on:
                - Concrete data points, statistics, and measurements
                - Specific claims and findings
                - Procedural steps and methods
                - Causal relationships
                - Definitions of technical terms
                
                For each fact, provide:
                1. The exact factual statement
                2. The immediate context where it appears
                3. What type of fact it is (numerical, procedural, causal, definitional)
                
                Document section:
                {chunk}
                
                Extract facts in this format:
                FACT: [exact factual statement]
                CONTEXT: [surrounding context]
                TYPE: [numerical/procedural/causal/definitional/categorical]
                CONFIDENCE: [high/medium/low]
                ---
                """
                
                model_spec = config.get("data_generation_model", "")
                response = await self.llm_manager.generate_response(model_spec, fact_extraction_prompt, config)
                
                chunk_facts = self._parse_structured_facts(response, doc_path, f"chunk_{i}")
                facts.extend(chunk_facts)
            
            return facts
            
        except Exception as e:
            logger.error(f"Structured fact extraction failed for {doc_path}: {str(e)}")
            return []
    
    async def _extract_structured_concepts(self, doc_path: str, config: Dict[str, Any]) -> List[ExtractedConcept]:
        """Extract structured, high-quality concepts from document"""
        try:
            content = self._read_document_content(doc_path)
            if not content:
                return []
            
            # Split content into chunks for better processing
            chunks = self._split_content_intelligently(content)
            concepts = []
            
            for i, chunk in enumerate(chunks[:5]):  # Process first 5 chunks
                concept_extraction_prompt = f"""
                Analyze this document section and identify key concepts, theories, methods, and important terms.
                
                For each concept, provide:
                1. The concept name (be specific and precise)
                2. A clear, concise definition
                3. Concrete examples or applications mentioned
                4. How it relates to other concepts in the text
                5. What domain/field it belongs to
                
                Document section:
                {chunk}
                
                Extract concepts in this format:
                CONCEPT: [specific concept name]
                DEFINITION: [clear, concise definition]
                EXAMPLES: [concrete examples from text]
                RELATIONSHIPS: [how it relates to other concepts]
                DOMAIN: [field/domain it belongs to]
                CONFIDENCE: [high/medium/low]
                ---
                """
                
                model_spec = config.get("data_generation_model", "")
                response = await self.llm_manager.generate_response(model_spec, concept_extraction_prompt, config)
                
                chunk_concepts = self._parse_structured_concepts(response, doc_path)
                concepts.extend(chunk_concepts)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Structured concept extraction failed for {doc_path}: {str(e)}")
            return []
    
    def _split_content_intelligently(self, content: str, chunk_size: int = 3000) -> List[str]:
        """Split content into intelligent chunks preserving context"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _parse_structured_facts(self, response: str, doc_path: str, location: str) -> List[ExtractedFact]:
        """Parse structured fact extraction response"""
        facts = []
        try:
            fact_blocks = response.split('---')
            
            for block in fact_blocks:
                if not block.strip():
                    continue
                
                fact_data = {}
                lines = block.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('FACT:'):
                        fact_data['content'] = line[5:].strip()
                    elif line.startswith('CONTEXT:'):
                        fact_data['context'] = line[8:].strip()
                    elif line.startswith('TYPE:'):
                        fact_data['fact_type'] = line[5:].strip()
                    elif line.startswith('CONFIDENCE:'):
                        confidence_str = line[11:].strip().lower()
                        fact_data['confidence'] = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(confidence_str, 0.5)
                
                if 'content' in fact_data and len(fact_data['content']) > 10:
                    fact = ExtractedFact(
                        content=fact_data.get('content', ''),
                        context=fact_data.get('context', ''),
                        confidence=fact_data.get('confidence', 0.5),
                        source_location=f"{doc_path}:{location}",
                        fact_type=fact_data.get('fact_type', 'general')
                    )
                    facts.append(fact)
            
        except Exception as e:
            logger.error(f"Failed to parse structured facts: {str(e)}")
        
        return facts
    
    def _parse_structured_concepts(self, response: str, doc_path: str) -> List[ExtractedConcept]:
        """Parse structured concept extraction response"""
        concepts = []
        try:
            concept_blocks = response.split('---')
            
            for block in concept_blocks:
                if not block.strip():
                    continue
                
                concept_data = {}
                lines = block.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('CONCEPT:'):
                        concept_data['name'] = line[8:].strip()
                    elif line.startswith('DEFINITION:'):
                        concept_data['definition'] = line[11:].strip()
                    elif line.startswith('EXAMPLES:'):
                        examples_str = line[9:].strip()
                        concept_data['examples'] = [ex.strip() for ex in examples_str.split(',') if ex.strip()]
                    elif line.startswith('RELATIONSHIPS:'):
                        relationships_str = line[14:].strip()
                        concept_data['relationships'] = [rel.strip() for rel in relationships_str.split(',') if rel.strip()]
                    elif line.startswith('DOMAIN:'):
                        concept_data['domain'] = line[7:].strip()
                    elif line.startswith('CONFIDENCE:'):
                        confidence_str = line[11:].strip().lower()
                        concept_data['confidence'] = {'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(confidence_str, 0.5)
                
                if 'name' in concept_data and 'definition' in concept_data:
                    concept = ExtractedConcept(
                        name=concept_data.get('name', ''),
                        definition=concept_data.get('definition', ''),
                        examples=concept_data.get('examples', []),
                        relationships=concept_data.get('relationships', []),
                        domain=concept_data.get('domain', 'general'),
                        confidence=concept_data.get('confidence', 0.5)
                    )
                    concepts.append(concept)
            
        except Exception as e:
            logger.error(f"Failed to parse structured concepts: {str(e)}")
        
        return concepts
    
    async def _generate_factual_qa_pairs(self, facts: List[ExtractedFact], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate high-quality factual Q&A pairs"""
        qa_pairs = []
        
        # Select high-confidence facts
        high_quality_facts = [f for f in facts if f.confidence >= 0.7]
        
        for fact in high_quality_facts[:25]:  # Limit to best facts
            try:
                # Choose appropriate template based on fact type
                if fact.fact_type == 'numerical':
                    templates = self.instruction_templates['factual_specific']
                else:
                    templates = self.instruction_templates['factual_direct']
                
                # Generate natural question
                question_prompt = f"""
                Create a natural, specific question that can be answered using this factual information:
                
                Fact: {fact.content}
                Context: {fact.context}
                
                The question should:
                - Be clear and specific
                - Sound natural (not template-like)
                - Be answerable with the given information
                - Test understanding of the fact
                
                Generate only the question, nothing else.
                """
                
                model_spec = config.get("data_generation_model", "")
                question = await self.llm_manager.generate_response(model_spec, question_prompt, config)
                question = self._clean_response(question)
                
                if not self._is_valid_instruction(question):
                    continue
                
                # Generate comprehensive answer
                answer_prompt = f"""
                Answer this question comprehensively using the provided information:
                
                Question: {question}
                
                Available information:
                - Fact: {fact.content}
                - Context: {fact.context}
                
                Provide a clear, informative answer that:
                - Directly answers the question
                - Uses the factual information provided
                - Includes relevant context
                - Is well-structured and complete
                
                Do not include disclaimers or meta-commentary.
                """
                
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = self._clean_response(answer)
                
                if self._is_valid_output(answer):
                    qa_pairs.append({
                        "instruction": question,
                        "input": "",
                        "output": answer
                    })
                
            except Exception as e:
                logger.error(f"Failed to generate factual Q&A: {str(e)}")
                continue
        
        return qa_pairs
    
    async def _generate_conceptual_qa_pairs(self, concepts: List[ExtractedConcept], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate high-quality conceptual Q&A pairs"""
        qa_pairs = []
        
        # Select high-confidence concepts
        high_quality_concepts = [c for c in concepts if c.confidence >= 0.7]
        
        for concept in high_quality_concepts[:20]:
            try:
                # Generate definition question
                question_prompt = f"""
                Create a natural question asking for an explanation of this concept:
                
                Concept: {concept.name}
                Domain: {concept.domain}
                
                The question should sound natural and conversational, not template-like.
                Generate only the question.
                """
                
                model_spec = config.get("data_generation_model", "")
                question = await self.llm_manager.generate_response(model_spec, question_prompt, config)
                question = self._clean_response(question)
                
                if not self._is_valid_instruction(question):
                    continue
                
                # Generate comprehensive answer
                answer_prompt = f"""
                Provide a comprehensive explanation of this concept:
                
                Question: {question}
                
                Concept information:
                - Name: {concept.name}
                - Definition: {concept.definition}
                - Examples: {', '.join(concept.examples) if concept.examples else 'None provided'}
                - Domain: {concept.domain}
                
                Your answer should:
                - Clearly explain what the concept is
                - Provide context and background
                - Include examples if available
                - Explain its significance or applications
                - Be informative and well-structured
                
                Do not include disclaimers or meta-commentary.
                """
                
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = self._clean_response(answer)
                
                if self._is_valid_output(answer):
                    qa_pairs.append({
                        "instruction": question,
                        "input": "",
                        "output": answer
                    })
                
            except Exception as e:
                logger.error(f"Failed to generate conceptual Q&A: {str(e)}")
                continue
        
        return qa_pairs
    
    async def _generate_analytical_qa_pairs(self, concepts: List[ExtractedConcept], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate analytical comparison Q&A pairs"""
        qa_pairs = []
        
        high_quality_concepts = [c for c in concepts if c.confidence >= 0.7]
        
        # Generate comparison questions between related concepts
        for i in range(min(10, len(high_quality_concepts))):
            for j in range(i + 1, min(i + 3, len(high_quality_concepts))):
                try:
                    concept1 = high_quality_concepts[i]
                    concept2 = high_quality_concepts[j]
                    
                    # Skip if concepts are too similar or unrelated
                    if concept1.domain != concept2.domain and concept1.domain != 'general' and concept2.domain != 'general':
                        continue
                    
                    question_prompt = f"""
                    Create a thoughtful analytical question that compares or relates these two concepts:
                    
                    Concept 1: {concept1.name}
                    Concept 2: {concept2.name}
                    
                    The question should:
                    - Ask for analysis or comparison
                    - Be intellectually engaging
                    - Sound natural and conversational
                    
                    Generate only the question.
                    """
                    
                    model_spec = config.get("data_generation_model", "")
                    question = await self.llm_manager.generate_response(model_spec, question_prompt, config)
                    question = self._clean_response(question)
                    
                    if not self._is_valid_instruction(question):
                        continue
                    
                    # Generate analytical answer
                    answer_prompt = f"""
                    Provide a thoughtful analytical answer to this question:
                    
                    Question: {question}
                    
                    Concept 1: {concept1.name}
                    - Definition: {concept1.definition}
                    - Examples: {', '.join(concept1.examples) if concept1.examples else 'None'}
                    
                    Concept 2: {concept2.name}
                    - Definition: {concept2.definition}
                    - Examples: {', '.join(concept2.examples) if concept2.examples else 'None'}
                    
                    Your answer should:
                    - Compare and contrast the concepts
                    - Analyze their relationships
                    - Discuss similarities and differences
                    - Provide insights and implications
                    - Be well-structured and analytical
                    
                    Do not include disclaimers or meta-commentary.
                    """
                    
                    answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                    answer = self._clean_response(answer)
                    
                    if self._is_valid_output(answer):
                        qa_pairs.append({
                            "instruction": question,
                            "input": "",
                            "output": answer
                        })
                
                except Exception as e:
                    logger.error(f"Failed to generate analytical Q&A: {str(e)}")
                    continue
        
        return qa_pairs
    
    async def _generate_application_qa_pairs(self, concepts: List[ExtractedConcept], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate application-focused Q&A pairs"""
        qa_pairs = []
        
        high_quality_concepts = [c for c in concepts if c.confidence >= 0.7 and c.examples]
        
        for concept in high_quality_concepts[:15]:
            try:
                question_prompt = f"""
                Create a practical question about real-world applications of this concept:
                
                Concept: {concept.name}
                Domain: {concept.domain}
                Examples: {', '.join(concept.examples)}
                
                The question should:
                - Ask about practical applications or use cases
                - Be relevant to the concept's domain
                - Sound natural and engaging
                
                Generate only the question.
                """
                
                model_spec = config.get("data_generation_model", "")
                question = await self.llm_manager.generate_response(model_spec, question_prompt, config)
                question = self._clean_response(question)
                
                if not self._is_valid_instruction(question):
                    continue
                
                # Generate practical answer
                answer_prompt = f"""
                Provide a practical, informative answer about applications:
                
                Question: {question}
                
                Concept: {concept.name}
                Definition: {concept.definition}
                Examples: {', '.join(concept.examples)}
                Domain: {concept.domain}
                
                Your answer should:
                - Explain practical applications and use cases
                - Provide concrete examples
                - Discuss benefits and implementation
                - Be actionable and informative
                - Connect theory to practice
                
                Do not include disclaimers or meta-commentary.
                """
                
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = self._clean_response(answer)
                
                if self._is_valid_output(answer):
                    qa_pairs.append({
                        "instruction": question,
                        "input": "",
                        "output": answer
                    })
                
            except Exception as e:
                logger.error(f"Failed to generate application Q&A: {str(e)}")
                continue
        
        return qa_pairs
    
    async def _generate_document_qa_pairs(self, document_paths: List[str], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate Q&A pairs that require document-level understanding"""
        qa_pairs = []
        
        document_questions = [
            "What are the main findings or conclusions presented in this document?",
            "What evidence or data supports the key arguments?",
            "What methodology or approach is described?",
            "What are the practical implications of the information presented?",
            "What recommendations or next steps are suggested?",
            "What problems or challenges are identified?",
            "What solutions or approaches are proposed?",
            "How do the different sections or topics relate to each other?"
        ]
        
        for question in document_questions[:6]:  # Limit to avoid too many
            try:
                # Use RAG to get relevant context
                relevant_chunks = await self.rag_system.retrieve_relevant_chunks(question, top_k=5)
                
                if not relevant_chunks:
                    continue
                
                # Combine and clean context
                context_texts = []
                for chunk in relevant_chunks:
                    text = chunk.get("text", "").strip()
                    if text and len(text) > 50:  # Only include substantial chunks
                        context_texts.append(text)
                
                if not context_texts:
                    continue
                
                combined_context = "\n\n".join(context_texts[:3])  # Use top 3 chunks
                
                # Generate comprehensive answer
                answer_prompt = f"""
                Answer this question based on the provided document information:
                
                Question: {question}
                
                Document information:
                {combined_context}
                
                Provide a comprehensive, well-structured answer that:
                - Directly addresses the question
                - Uses specific information from the documents
                - Is organized and easy to follow
                - Includes relevant details and examples
                - Synthesizes information effectively
                
                Do not include disclaimers or meta-commentary.
                """
                
                model_spec = config.get("data_generation_model", "")
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = self._clean_response(answer)
                
                if self._is_valid_output(answer):
                    qa_pairs.append({
                        "instruction": question,
                        "input": "Based on the provided document",
                        "output": answer
                    })
                
            except Exception as e:
                logger.error(f"Failed to generate document Q&A: {str(e)}")
                continue
        
        return qa_pairs
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize LLM response"""
        if not response:
            return ""
        
        # Remove thinking tags and other unwanted content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Remove disclaimers and meta-commentary
        for phrase in self.quality_filters["forbidden_phrases"]:
            response = response.replace(phrase, "")
        
        # Clean up formatting
        response = response.strip()
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)  # Remove excessive newlines
        response = re.sub(r'^\s*[-*]\s*', '', response, flags=re.MULTILINE)  # Remove bullet points at start
        
        return response
    
    def _is_valid_instruction(self, instruction: str) -> bool:
        """Check if instruction meets quality criteria"""
        if not instruction or len(instruction) < self.quality_filters["min_instruction_length"]:
            return False
        
        # Check for forbidden phrases
        instruction_lower = instruction.lower()
        for phrase in self.quality_filters["forbidden_phrases"]:
            if phrase.lower() in instruction_lower:
                return False
        
        # Check if it's actually a question or instruction
        if not (instruction.endswith('?') or any(word in instruction_lower for word in 
                ['what', 'how', 'why', 'when', 'where', 'explain', 'describe', 'define', 'analyze', 'compare'])):
            return False
        
        return True
    
    def _is_valid_output(self, output: str) -> bool:
        """Check if output meets quality criteria"""
        if not output:
            return False
        
        output_len = len(output)
        if output_len < self.quality_filters["min_output_length"] or output_len > self.quality_filters["max_output_length"]:
            return False
        
        # Check for forbidden phrases
        output_lower = output.lower()
        for phrase in self.quality_filters["forbidden_phrases"]:
            if phrase.lower() in output_lower:
                return False
        
        # Check content quality - should not be mostly repetitive
        words = output.split()
        if len(words) < 10:
            return False
        
        # Check for excessive repetition
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        if repetition_ratio > self.quality_filters["max_repetition_ratio"]:
            return False
        
        return True
    
    def _apply_quality_filters(self, alpaca_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply comprehensive quality filtering"""
        high_quality_data = []
        
        for example in alpaca_data:
            try:
                instruction = example.get("instruction", "")
                output = example.get("output", "")
                
                # Apply validation checks
                if not self._is_valid_instruction(instruction):
                    continue
                
                if not self._is_valid_output(output):
                    continue
                
                # Additional quality checks
                if self._is_high_quality_pair(instruction, output):
                    high_quality_data.append(example)
                
            except Exception as e:
                logger.error(f"Error in quality filtering: {str(e)}")
                continue
        
        return high_quality_data
    
    def _is_high_quality_pair(self, instruction: str, output: str) -> bool:
        """Check if instruction-output pair is high quality"""
        # Check if output actually answers the instruction
        instruction_words = set(instruction.lower().split())
        output_words = set(output.lower().split())
        
        # Should have some overlap but not be identical
        overlap = len(instruction_words.intersection(output_words))
        overlap_ratio = overlap / len(instruction_words) if instruction_words else 0
        
        # Good answers have some overlap (0.1-0.7) but aren't just repetitions
        if overlap_ratio < 0.1 or overlap_ratio > 0.7:
            return False
        
        # Check if output is substantive
        sentences = output.split('.')
        if len(sentences) < 2:  # Should have at least 2 sentences
            return False
        
        return True
    
    def _balance_dataset(self, alpaca_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Balance dataset for diversity and remove duplicates"""
        if not alpaca_data:
            return []
        
        # Remove near-duplicates
        unique_data = self._remove_duplicates(alpaca_data)
        
        # Shuffle for diversity
        random.shuffle(unique_data)
        
        # Limit total size to prevent overwhelming datasets
        max_examples = 100  # Reasonable limit for quality over quantity
        if len(unique_data) > max_examples:
            unique_data = unique_data[:max_examples]
        
        return unique_data
    
    def _remove_duplicates(self, alpaca_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate and near-duplicate examples"""
        unique_data = []
        seen_instructions = set()
        seen_outputs = set()
        
        for example in alpaca_data:
            instruction = example.get("instruction", "").lower().strip()
            output = example.get("output", "").lower().strip()
            
            # Skip if we've seen very similar instruction or output
            instruction_key = self._normalize_text_for_comparison(instruction)
            output_key = self._normalize_text_for_comparison(output)
            
            if instruction_key in seen_instructions or output_key in seen_outputs:
                continue
            
            seen_instructions.add(instruction_key)
            seen_outputs.add(output_key)
            unique_data.append(example)
        
        return unique_data
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for duplicate detection"""
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Take first 50 characters for comparison
        return normalized[:50]
    
    def _calculate_quality_metrics(self, alpaca_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset"""
        if not alpaca_data:
            return {}
        
        total_examples = len(alpaca_data)
        
        # Calculate average lengths
        instruction_lengths = [len(ex.get("instruction", "")) for ex in alpaca_data]
        output_lengths = [len(ex.get("output", "")) for ex in alpaca_data]
        
        # Calculate diversity metrics
        unique_instructions = len(set(ex.get("instruction", "") for ex in alpaca_data))
        unique_outputs = len(set(ex.get("output", "") for ex in alpaca_data))
        
        return {
            "total_examples": total_examples,
            "avg_instruction_length": sum(instruction_lengths) / len(instruction_lengths),
            "avg_output_length": sum(output_lengths) / len(output_lengths),
            "instruction_diversity": unique_instructions / total_examples,
            "output_diversity": unique_outputs / total_examples,
            "min_instruction_length": min(instruction_lengths),
            "max_instruction_length": max(instruction_lengths),
            "min_output_length": min(output_lengths),
            "max_output_length": max(output_lengths)
        }
    
    def _read_document_content(self, doc_path: str) -> str:
        """Read document content based on file type"""
        try:
            if doc_path.lower().endswith('.pdf'):
                # Handle PDF files
                try:
                    import pypdf
                    with open(doc_path, 'rb') as f:
                        reader = pypdf.PdfReader(f)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() + "\n"
                    return content
                except ImportError:
                    logger.warning("pypdf not installed, trying PyPDF2")
                    try:
                        import PyPDF2
                        with open(doc_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            content = ""
                            for page in reader.pages:
                                content += page.extract_text() + "\n"
                        return content
                    except ImportError:
                        logger.warning("PyPDF2 not installed, trying PyMuPDF")
                        try:
                            import fitz  # PyMuPDF
                            doc = fitz.open(doc_path)
                            content = ""
                            for page in doc:
                                content += page.get_text() + "\n"
                            doc.close()
                            return content
                        except ImportError:
                            logger.error("No PDF reading library available (pypdf, PyPDF2, or PyMuPDF)")
                            return ""
                except Exception as e:
                    logger.error(f"Error reading PDF {doc_path}: {str(e)}")
                    return ""
            
            elif doc_path.lower().endswith('.csv'):
                # Handle CSV files
                try:
                    import pandas as pd
                    df = pd.read_csv(doc_path)
                    return df.to_string()
                except ImportError:
                    # Fallback to basic CSV reading
                    import csv
                    content = ""
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            content += ", ".join(row) + "\n"
                    return content
                except Exception as e:
                    logger.error(f"Error reading CSV {doc_path}: {str(e)}")
                    return ""
            
            else:
                # Handle text files
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(doc_path, 'r', encoding='latin-1') as f:
                            return f.read()
                    except Exception as e:
                        logger.error(f"Error reading text file {doc_path}: {str(e)}")
                        return ""
                except Exception as e:
                    logger.error(f"Error reading file {doc_path}: {str(e)}")
                    return ""
        
        except Exception as e:
            logger.error(f"Error reading document {doc_path}: {str(e)}")
            return ""
    
    def save_alpaca_dataset(self, alpaca_data: List[Dict[str, str]], output_path: str) -> bool:
        """Save Alpaca dataset to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Improved Alpaca dataset saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Alpaca dataset: {str(e)}")
            return False
