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
                
                Please extract facts in JSON format, adhering to the following schema. Ensure all facts are included and are highly specific.
                
                JSON Schema for Facts:
                [
                    {{
                        "content": "The exact factual statement.",
                        "context": "The immediate context where it appears.",
                        "fact_type": "numerical/procedural/causal/definitional/categorical/general",
                        "confidence": "high/medium/low"
                    }},
                    // ... more facts
                ]
                
                Example:
                [
                    {{
                        "content": "The Earth's circumference at the equator is approximately 40,075 kilometers.",
                        "context": "The document states, 'The Earth's circumference at the equator is approximately 40,075 kilometers, a key measurement in geodesy.'",
                        "fact_type": "numerical",
                        "confidence": "high"
                    }},
                    {{
                        "content": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
                        "context": "In biology, photosynthesis is defined as the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
                        "fact_type": "definitional",
                        "confidence": "high"
                    }}
                ]
                """
                
                model_spec = config.get("data_generation_model", "")
                response = await self.llm_manager.generate_response(model_spec, fact_extraction_prompt, config)
                
                try:
                    chunk_facts_data = json.loads(response)
                    chunk_facts = [ExtractedFact(**f) for f in chunk_facts_data]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON facts from LLM response: {e}. Response: {response[:500]}")
                    chunk_facts = [] # Fallback to empty list if JSON parsing fails
                
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
            
            for i, chunk in enumerate(chunks):  # Process all chunks, no hard limit
                concept_extraction_prompt = f"""
                Analyze this document section and identify all key concepts, theories, methods, and important terms.
                
                Please extract concepts in JSON format, adhering to the following schema. Ensure all relevant concepts are included.
                
                JSON Schema for Concepts:
                [
                    {{
                        "name": "Specific concept name.",
                        "definition": "A clear, concise definition.",
                        "examples": ["Concrete example 1", "Concrete example 2"],
                        "relationships": ["How it relates to other concepts in the text"],
                        "domain": "Field/domain it belongs to",
                        "confidence": "high/medium/low"
                    }},
                    // ... more concepts
                ]
                
                Example:
                [
                    {{
                        "name": "Quantum Entanglement",
                        "definition": "A phenomenon where two or more particles become linked in such a way that they share the same fate, regardless of distance.",
                        "examples": ["Bell test experiments", "Quantum cryptography"],
                        "relationships": ["Related to quantum mechanics", "Contrasts with classical physics"],
                        "domain": "Physics",
                        "confidence": "high"
                    }},
                    {{
                        "name": "Supply Chain Management",
                        "definition": "The oversight of materials, information, and finances as they move in a process from supplier to manufacturer to wholesaler to retailer to consumer.",
                        "examples": ["Just-in-time inventory", "Logistics optimization"],
                        "relationships": ["Interacts with operations management", "Impacts business strategy"],
                        "domain": "Business",
                        "confidence": "high"
                    }}
                ]
                
                Document section:
                {chunk}
                """
                
                model_spec = config.get("data_generation_model", "")
                response = await self.llm_manager.generate_response(model_spec, concept_extraction_prompt, config)
                
                try:
                    chunk_concepts_data = json.loads(response)
                    chunk_concepts = [ExtractedConcept(**c) for c in chunk_concepts_data]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON concepts from LLM response: {e}. Response: {response[:500]}")
                    chunk_concepts = [] # Fallback to empty list if JSON parsing fails
                
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
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extracts a JSON object or list from a string, even with surrounding text or markdown."""
        # Find the start of the JSON - either { or [
        start_brace = response.find('{')
        start_bracket = response.find('[')

        if start_brace == -1 and start_bracket == -1:
            return None

        if start_brace == -1:
            start = start_bracket
        elif start_bracket == -1:
            start = start_brace
        else:
            start = min(start_brace, start_bracket)

        # Find the corresponding end of the JSON
        if response[start] == '{':
            end_char = '}'
            level = 1
        else:
            end_char = ']'
            level = 1
        
        end = -1
        for i in range(start + 1, len(response)):
            if response[i] == response[start]:
                level += 1
            elif response[i] == end_char:
                level -= 1
                if level == 0:
                    end = i + 1
                    break
        
        if end == -1:
            return None

        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _parse_structured_facts(self, response: str, doc_path: str, location: str) -> List[ExtractedFact]:
        """Parse structured fact extraction response (now expects JSON)"""
        facts = []
        try:
            data = self._extract_json_from_response(response)
            if not data or not isinstance(data, list):
                logger.error(f"No valid JSON list found in fact extraction response: {response[:500]}")
                return []

            for item in data:
                if isinstance(item, dict) and 'content' in item and len(item['content']) > 10:
                    fact = ExtractedFact(
                        content=item.get('content', ''),
                        context=item.get('context', ''),
                        confidence={'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(str(item.get('confidence', 'medium')).lower(), 0.5),
                        source_location=f"{doc_path}:{location}",
                        fact_type=item.get('fact_type', 'general')
                    )
                    facts.append(fact)
        except Exception as e:
            logger.error(f"Failed to parse structured facts: {str(e)}")
        
        return facts
    
    def _parse_structured_concepts(self, response: str, doc_path: str) -> List[ExtractedConcept]:
        """Parse structured concept extraction response (now expects JSON)"""
        concepts = []
        try:
            data = self._extract_json_from_response(response)
            if not data or not isinstance(data, list):
                logger.error(f"No valid JSON list found in concept extraction response: {response[:500]}")
                return []

            for item in data:
                if isinstance(item, dict) and 'name' in item and 'definition' in item:
                    concept = ExtractedConcept(
                        name=item.get('name', ''),
                        definition=item.get('definition', ''),
                        examples=item.get('examples', []),
                        relationships=item.get('relationships', []),
                        domain=item.get('domain', 'general'),
                        confidence={'high': 0.9, 'medium': 0.7, 'low': 0.5}.get(str(item.get('confidence', 'medium')).lower(), 0.5)
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
        
        for fact in high_quality_facts:  # Removed hard limit
            try:
                # Choose appropriate template based on fact type
                if fact.fact_type == 'numerical':
                    templates = self.instruction_templates['factual_specific']
                else:
                    templates = self.instruction_templates['factual_direct']
                
                # Generate natural question
                question_prompt = f"""
                Create a natural, specific question that can be answered using this factual information.
                The question should be clear, specific, sound natural, be answerable with the given information, and test understanding of the fact.
                
                Fact: "{fact.content}"
                Context: "{fact.context}"
                
                Example Question: "What is the capital of France?"
                Example Answer: "The capital of France is Paris."
                
                Generate only the question, nothing else.
                """
                
                model_spec = config.get("data_generation_model", "")
                question, answer = await self._generate_qa_with_retries(model_spec, question_prompt, f"""
                Answer this question comprehensively using the provided information.
                The answer should directly answer the question, use the factual information provided, include relevant context, and be well-structured and complete.
                Do not include disclaimers or meta-commentary.
                
                Question: {{question}}
                
                Available information:
                - Fact: {fact.content}
                - Context: {fact.context}
                
                Example Question: "What is the capital of France?"
                Example Answer: "The capital of France is Paris, a major European city known for its art and culture."
                """, config)

                if question and answer:
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
        
        for concept in high_quality_concepts: # Removed hard limit
            try:
                # Generate definition question
                question_prompt = f"""
                Create a natural question asking for an explanation of this concept.
                The question should sound natural and conversational, not template-like.
                Generate only the question.
                
                Concept: {concept.name}
                Domain: {concept.domain}
                
                Example Question: "Can you explain the concept of 'Photosynthesis'?"
                """
                
                model_spec = config.get("data_generation_model", "")
                question, answer = await self._generate_qa_with_retries(model_spec, question_prompt, f"""
                Provide a comprehensive explanation of this concept.
                Your answer should clearly explain what the concept is, provide context and background, include examples if available, explain its significance or applications, and be informative and well-structured.
                Do not include disclaimers or meta-commentary.
                
                Question: {{question}}
                
                Concept information:
                - Name: {concept.name}
                - Definition: {concept.definition}
                - Examples: {', '.join(concept.examples) if concept.examples else 'None provided'}
                - Domain: {concept.domain}
                
                Example Answer: "Photosynthesis is the process used by plants, algae and certain bacteria to turn sunlight into chemical energy. This energy is then used to fuel the organism's activities. It is a vital process for life on Earth as it produces oxygen as a byproduct."
                """, config)

                if question and answer:
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
        for i in range(len(high_quality_concepts)): # Removed hard limit
            for j in range(i + 1, len(high_quality_concepts)): # Removed hard limit
                try:
                    concept1 = high_quality_concepts[i]
                    concept2 = high_quality_concepts[j]
                    
                    # Skip if concepts are too similar or unrelated
                    if concept1.domain != concept2.domain and concept1.domain != 'general' and concept2.domain != 'general':
                        continue
                    
                    question_prompt = f"""
                    Create a thoughtful analytical question that compares or relates these two concepts.
                    The question should ask for analysis or comparison, be intellectually engaging, and sound natural and conversational.
                    Generate only the question.
                    
                    Concept 1: {concept1.name}
                    Concept 2: {concept2.name}
                    
                    Example Question: "Compare and contrast 'Artificial Intelligence' and 'Machine Learning'."
                    """
                    
                    model_spec = config.get("data_generation_model", "")
                    question, answer = await self._generate_qa_with_retries(model_spec, question_prompt, f"""
                    Provide a thoughtful analytical answer to this question.
                    Your answer should compare and contrast the concepts, analyze their relationships, discuss similarities and differences, provide insights and implications, and be well-structured and analytical.
                    Do not include disclaimers or meta-commentary.
                    
                    Question: {{question}}
                    
                    Concept 1: {concept1.name}
                    - Definition: {concept1.definition}
                    - Examples: {', '.join(concept1.examples) if concept1.examples else 'None'}
                    
                    Concept 2: {concept2.name}
                    - Definition: {concept2.definition}
                    - Examples: {', '.join(concept2.examples) if concept2.examples else 'None'}
                    
                    Example Answer: "Artificial Intelligence (AI) is a broad field focused on creating machines that can perform tasks requiring human intelligence, while Machine Learning (ML) is a subset of AI that enables systems to learn from data without explicit programming. ML is a method to achieve AI."
                    """, config)

                    if question and answer:
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
        
        for concept in high_quality_concepts: # Removed hard limit
            try:
                question_prompt = f"""
                Create a practical question about real-world applications of this concept.
                The question should ask about practical applications or use cases, be relevant to the concept's domain, and sound natural and engaging.
                Generate only the question.
                
                Concept: {concept.name}
                Domain: {concept.domain}
                Examples: {', '.join(concept.examples)}
                
                Example Question: "How is 'Blockchain' applied in finance?"
                """
                
                model_spec = config.get("data_generation_model", "")
                question, answer = await self._generate_qa_with_retries(model_spec, question_prompt, f"""
                Provide a practical, informative answer about applications.
                Your answer should explain practical applications and use cases, provide concrete examples, discuss benefits and implementation, be actionable and informative, and connect theory to practice.
                Do not include disclaimers or meta-commentary.
                
                Question: {{question}}
                
                Concept: {concept.name}
                Definition: {concept.definition}
                Examples: {', '.join(concept.examples)}
                Domain: {concept.domain}
                
                Example Answer: "Blockchain is applied in finance for secure and transparent transactions, such as in cryptocurrencies like Bitcoin, and for smart contracts that automate agreements without intermediaries."
                """, config)

                if question and answer:
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
        
        for question in document_questions:  # Removed hard limit
            try:
                # Use RAG to get relevant context
                relevant_chunks = await self.rag_system.retrieve_relevant_chunks(question, top_k=5)
                
                if not relevant_chunks:
                    logger.debug(f"Document Q&A: No relevant chunks found for question: {question}")
                    continue
                
                # Combine and clean context
                context_texts = []
                for chunk in relevant_chunks:
                    text = chunk.get("text", "").strip()
                    if text and len(text) > 50:  # Only include substantial chunks
                        context_texts.append(text)
                
                if not context_texts:
                    logger.debug(f"Document Q&A: No substantial context texts found for question: {question}")
                    continue
                
                combined_context = "\n\n".join(context_texts) # Use all relevant chunks
                
                # Generate comprehensive answer
                answer_prompt = f"""
                Answer this question based on the provided document information.
                Your answer should directly address the question, use specific information from the documents, be organized and easy to follow, include relevant details and examples, and synthesize information effectively.
                Do not include disclaimers or meta-commentary.
                
                Question: {question}
                
                Document information:
                {combined_context}
                
                Example Question: "Summarize the key findings from the document."
                Example Answer: "The document's key findings indicate a significant correlation between X and Y, suggesting a novel approach to Z, as evidenced by the experimental data presented in Section 3.2."
                """
                
                model_spec = config.get("data_generation_model", "")
                # For document-level questions, we can use a simplified retry logic as it's less about refining a specific fact
                # and more about summarizing broader context.
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = self._clean_response(answer)
                
                if self._is_valid_output(answer):
                    qa_pairs.append({
                        "instruction": question,
                        "input": "Based on the provided documents",
                        "output": answer
                    })
                else:
                    logger.debug(f"Document Q&A: Output '{answer}' failed validation for instruction '{question}'.")
                
            except Exception as e:
                logger.error(f"Failed to generate document Q&A: {str(e)}")
                continue
        
        return qa_pairs
    
    async def _generate_qa_with_retries(self, model_spec: str, question_prompt: str, answer_prompt: str, config: Dict[str, Any], max_retries: int = 2) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates Q&A with retries and iterative refinement.
        Returns (question, answer) or (None, None) if unsuccessful.
        """
        question = None
        answer = None
        
        for attempt in range(max_retries + 1):
            try:
                # Generate question
                current_question_prompt = question_prompt
                if question: # If re-attempting, provide previous question for context
                    current_question_prompt += f"\n\nPrevious attempt's question: {question}"
                
                question = await self.llm_manager.generate_response(model_spec, current_question_prompt, config)
                question = self._clean_response(question)
                
                if not self._is_valid_instruction(question):
                    logger.debug(f"Attempt {attempt+1}: Instruction '{question}' failed validation. Retrying...")
                    continue # Try again
                
                # Generate answer
                current_answer_prompt = answer_prompt
                if answer: # If re-attempting, provide previous answer for context
                    current_answer_prompt += f"\n\nPrevious attempt's answer: {answer}"
                
                answer = await self.llm_manager.generate_response(model_spec, current_answer_prompt, config)
                answer = self._clean_response(answer)
                
                if self._is_valid_output(answer):
                    return question, answer # Success
                else:
                    logger.debug(f"Attempt {attempt+1}: Output '{answer}' failed validation for instruction '{question}'. Retrying...")
                    # Provide feedback to the LLM for the next attempt
                    feedback_prompt = f"The previous answer was not satisfactory. It failed the following quality checks: "
                    if not self._is_valid_output(answer):
                        feedback_prompt += "Output is too short or too long, or contains forbidden phrases, or is too repetitive. "
                    # Add more specific feedback based on _is_valid_output checks if needed
                    
                    question_prompt += f"\n\nManager feedback for next attempt: {feedback_prompt} Please improve the quality of the generated question."
                    answer_prompt += f"\n\nManager feedback for next attempt: {feedback_prompt} Please improve the quality of the generated answer."
                    
            except Exception as e:
                logger.error(f"Error during Q&A generation attempt {attempt+1}: {str(e)}")
                question, answer = None, None # Reset for next attempt
                
        logger.warning(f"Failed to generate valid Q&A after {max_retries+1} attempts for fact/concept.")
        return None, None # Failed after all retries
    
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
            logger.debug(f"Instruction '{instruction}' failed: too short ({len(instruction)} < {self.quality_filters['min_instruction_length']})")
            return False
        
        # Check for forbidden phrases
        instruction_lower = instruction.lower()
        for phrase in self.quality_filters["forbidden_phrases"]:
            if phrase.lower() in instruction_lower:
                logger.debug(f"Instruction '{instruction}' failed: contains forbidden phrase '{phrase}'")
                return False
        
        # Check if it's actually a question or instruction
        if not (instruction.endswith('?') or any(word in instruction_lower for word in 
                ['what', 'how', 'why', 'when', 'where', 'explain', 'describe', 'define', 'analyze', 'compare'])):
            logger.debug(f"Instruction '{instruction}' failed: not a question/instruction.")
            return False
        
        return True
    
    def _is_valid_output(self, output: str) -> bool:
        """Check if output meets quality criteria"""
        if not output:
            logger.debug("Output failed: empty.")
            return False
        
        output_len = len(output)
        if output_len < self.quality_filters["min_output_length"]:
            logger.debug(f"Output '{output}' failed: too short ({output_len} < {self.quality_filters['min_output_length']})")
            return False
        if output_len > self.quality_filters["max_output_length"]:
            logger.debug(f"Output '{output}' failed: too long ({output_len} > {self.quality_filters['max_output_length']})")
            return False
        
        # Check for forbidden phrases
        output_lower = output.lower()
        for phrase in self.quality_filters["forbidden_phrases"]:
            if phrase.lower() in output_lower:
                logger.debug(f"Output '{output}' failed: contains forbidden phrase '{phrase}'")
                return False
        
        # Check content quality - should not be mostly repetitive
        words = output.split()
        if len(words) < 10:
            logger.debug(f"Output '{output}' failed: too few words ({len(words)} < 10).")
            return False
        
        # Check for excessive repetition
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        if repetition_ratio > self.quality_filters["max_repetition_ratio"]:
            logger.debug(f"Output '{output}' failed: excessive repetition ({repetition_ratio:.2f} > {self.quality_filters['max_repetition_ratio']}).")
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
                    logger.debug(f"Example filtered (instruction invalid): {instruction}")
                    continue
                
                if not self._is_valid_output(output):
                    logger.debug(f"Example filtered (output invalid): {output}")
                    continue
                
                # Additional quality checks
                if self._is_high_quality_pair(instruction, output):
                    high_quality_data.append(example)
                else:
                    logger.debug(f"Example filtered (high quality pair check failed): Instruction: '{instruction}', Output: '{output}'")
                
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
        # Removed upper bound to allow direct factual answers
        if overlap_ratio < 0.1:
            logger.debug(f"High quality pair check failed: low overlap ratio ({overlap_ratio:.2f} < 0.1). Instruction: '{instruction}', Output: '{output}'")
            return False
        
        # Removed sentence count check to allow valid single-sentence answers
        
        return True
    
    def _balance_dataset(self, alpaca_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Balance dataset for diversity and remove duplicates"""
        if not alpaca_data:
            return []
        
        # Remove near-duplicates
        unique_data = self._remove_duplicates(alpaca_data)
        
        # Shuffle for diversity
        random.shuffle(unique_data)
        
        # Removed hard limit on total size to ensure all quality data is kept
        # max_examples = 100  # Reasonable limit for quality over quantity
        # if len(unique_data) > max_examples:
        #     unique_data = unique_data[:max_examples]
        
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
                logger.debug(f"Duplicate example removed: Instruction: '{instruction}', Output: '{output}'")
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
    
    def _calculate_quality_metrics(self, alpaca_data: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            "avg_instruction_length": sum(instruction_lengths) / total_examples if total_examples > 0 else 0,
            "avg_output_length": sum(output_lengths) / total_examples if total_examples > 0 else 0,
            "instruction_diversity": unique_instructions / total_examples if total_examples > 0 else 0,
            "output_diversity": unique_outputs / total_examples if total_examples > 0 else 0,
            "min_instruction_length": min(instruction_lengths) if instruction_lengths else 0,
            "max_instruction_length": max(instruction_lengths) if instruction_lengths else 0,
            "min_output_length": min(output_lengths) if output_lengths else 0,
            "max_output_length": max(output_lengths) if output_lengths else 0
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
