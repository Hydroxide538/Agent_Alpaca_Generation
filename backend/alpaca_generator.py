import json
import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import re
import random

logger = logging.getLogger(__name__)

class AlpacaFormatGenerator:
    """Generate training data in Alpaca format from documents using RAG and LLM"""
    
    def __init__(self, llm_manager, rag_system):
        self.llm_manager = llm_manager
        self.rag_system = rag_system
        self.question_templates = self._load_question_templates()
    
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load various question templates for different types of questions"""
        return {
            "factual": [
                "What is {concept}?",
                "Define {concept}.",
                "Explain {concept}.",
                "What does {concept} mean?",
                "Describe {concept}.",
                "What are the key characteristics of {concept}?",
                "How would you define {concept}?",
                "What is the meaning of {concept}?"
            ],
            "analytical": [
                "How does {concept1} relate to {concept2}?",
                "What is the relationship between {concept1} and {concept2}?",
                "Compare {concept1} and {concept2}.",
                "What are the differences between {concept1} and {concept2}?",
                "How do {concept1} and {concept2} interact?",
                "What is the connection between {concept1} and {concept2}?",
                "Analyze the relationship between {concept1} and {concept2}."
            ],
            "application": [
                "How can {concept} be applied in practice?",
                "What are the practical applications of {concept}?",
                "How would you use {concept} in a real-world scenario?",
                "Give an example of how {concept} is used.",
                "What are some use cases for {concept}?",
                "How can {concept} be implemented?",
                "What are the benefits of using {concept}?"
            ],
            "procedural": [
                "How do you {process}?",
                "What are the steps to {process}?",
                "Explain the process of {process}.",
                "What is the procedure for {process}?",
                "How would you go about {process}?",
                "What is involved in {process}?",
                "Describe how to {process}."
            ],
            "causal": [
                "What causes {effect}?",
                "Why does {effect} occur?",
                "What leads to {effect}?",
                "What are the reasons for {effect}?",
                "What factors contribute to {effect}?",
                "How does {cause} result in {effect}?",
                "What is the cause of {effect}?"
            ],
            "evaluative": [
                "What are the advantages and disadvantages of {concept}?",
                "Evaluate the effectiveness of {concept}.",
                "What are the pros and cons of {concept}?",
                "Assess the value of {concept}.",
                "What are the strengths and weaknesses of {concept}?",
                "How effective is {concept}?",
                "What are the benefits and drawbacks of {concept}?"
            ]
        }
    
    async def generate_alpaca_dataset(self, document_paths: List[str], config: Dict[str, Any], websocket_manager=None) -> Dict[str, Any]:
        """Generate complete Alpaca format dataset from documents"""
        try:
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Starting Alpaca format dataset generation..."
                })
            
            # Step 1: Process documents with RAG
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Processing documents and creating embeddings..."
                })
            
            rag_results = await self.rag_system.process_documents(document_paths, websocket_manager)
            
            # Step 2: Extract facts and concepts from each document
            all_facts = []
            all_concepts = []
            
            for doc_path in document_paths:
                if websocket_manager:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "info",
                        "message": f"Analyzing document: {doc_path}"
                    })
                
                doc_facts = await self._extract_facts_from_document(doc_path, config)
                doc_concepts = await self._extract_concepts_from_document(doc_path, config)
                
                all_facts.extend(doc_facts)
                all_concepts.extend(doc_concepts)
            
            # Step 3: Generate Q&A pairs in Alpaca format
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Generating Q&A pairs in Alpaca format..."
                })
            
            alpaca_data = []
            
            # Generate different types of questions
            alpaca_data.extend(await self._generate_factual_questions(all_facts, config))
            alpaca_data.extend(await self._generate_analytical_questions(all_concepts, config))
            alpaca_data.extend(await self._generate_application_questions(all_concepts, config))
            alpaca_data.extend(await self._generate_rag_based_questions(document_paths, config))
            
            # Step 4: Validate and clean the dataset
            validated_data = self._validate_alpaca_format(alpaca_data)
            
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"Generated {len(validated_data)} Alpaca format training examples"
                })
            
            return {
                "alpaca_data": validated_data,
                "statistics": {
                    "total_examples": len(validated_data),
                    "documents_processed": len(document_paths),
                    "facts_extracted": len(all_facts),
                    "concepts_extracted": len(all_concepts),
                    "rag_chunks": rag_results.get("total_chunks", 0)
                },
                "metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": config.get("data_generation_model"),
                    "embedding_model": config.get("embedding_model"),
                    "reranking_model": config.get("reranking_model")
                }
            }
            
        except Exception as e:
            logger.error(f"Alpaca dataset generation failed: {str(e)}")
            raise
    
    async def _extract_facts_from_document(self, doc_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract factual information from a document"""
        try:
            # Read document content based on file type
            content = self._read_document_content(doc_path)
            if not content:
                logger.warning(f"Could not read content from {doc_path}")
                return []
            
            # Use LLM to extract facts
            fact_extraction_prompt = f"""
            Analyze the following document and extract all factual information, data points, and key statements.
            For each fact, provide:
            1. The fact itself
            2. The context in which it appears
            3. Any supporting details
            
            Document content:
            {content[:4000]}  # Limit content to avoid token limits
            
            Please extract facts in a structured format.
            """
            
            model_spec = config.get("data_generation_model", "")
            facts_response = await self.llm_manager.generate_response(model_spec, fact_extraction_prompt, config)
            
            # Parse the response to extract structured facts
            facts = self._parse_facts_response(facts_response, doc_path)
            
            return facts
            
        except Exception as e:
            logger.error(f"Fact extraction failed for {doc_path}: {str(e)}")
            return []
    
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
    
    async def _extract_concepts_from_document(self, doc_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key concepts and topics from a document"""
        try:
            # Read document content based on file type
            content = self._read_document_content(doc_path)
            if not content:
                logger.warning(f"Could not read content from {doc_path}")
                return []
            
            # Use LLM to extract concepts
            concept_extraction_prompt = f"""
            Analyze the following document and identify all key concepts, topics, and important terms.
            For each concept, provide:
            1. The concept name
            2. A brief definition or description
            3. How it relates to other concepts in the document
            
            Document content:
            {content[:4000]}  # Limit content to avoid token limits
            
            Please extract concepts in a structured format.
            """
            
            model_spec = config.get("data_generation_model", "")
            concepts_response = await self.llm_manager.generate_response(model_spec, concept_extraction_prompt, config)
            
            # Parse the response to extract structured concepts
            concepts = self._parse_concepts_response(concepts_response, doc_path)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Concept extraction failed for {doc_path}: {str(e)}")
            return []
    
    def _parse_facts_response(self, response: str, doc_path: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract structured facts"""
        facts = []
        try:
            # Simple parsing - in a production system, you'd use more sophisticated parsing
            lines = response.split('\n')
            current_fact = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_fact:
                        current_fact["source_document"] = doc_path
                        facts.append(current_fact)
                        current_fact = {}
                    continue
                
                if line.startswith(('1.', '2.', '3.', '-', '*')):
                    # New fact
                    if current_fact:
                        current_fact["source_document"] = doc_path
                        facts.append(current_fact)
                    
                    current_fact = {
                        "fact": line,
                        "context": "",
                        "details": ""
                    }
                elif current_fact:
                    # Add to current fact
                    if "context" not in current_fact or not current_fact["context"]:
                        current_fact["context"] = line
                    else:
                        current_fact["details"] += " " + line
            
            # Add the last fact
            if current_fact:
                current_fact["source_document"] = doc_path
                facts.append(current_fact)
            
        except Exception as e:
            logger.error(f"Failed to parse facts response: {str(e)}")
        
        return facts
    
    def _parse_concepts_response(self, response: str, doc_path: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract structured concepts"""
        concepts = []
        try:
            # Simple parsing - in a production system, you'd use more sophisticated parsing
            lines = response.split('\n')
            current_concept = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_concept:
                        current_concept["source_document"] = doc_path
                        concepts.append(current_concept)
                        current_concept = {}
                    continue
                
                if line.startswith(('1.', '2.', '3.', '-', '*')):
                    # New concept
                    if current_concept:
                        current_concept["source_document"] = doc_path
                        concepts.append(current_concept)
                    
                    current_concept = {
                        "concept": line,
                        "definition": "",
                        "relationships": ""
                    }
                elif current_concept:
                    # Add to current concept
                    if "definition" not in current_concept or not current_concept["definition"]:
                        current_concept["definition"] = line
                    else:
                        current_concept["relationships"] += " " + line
            
            # Add the last concept
            if current_concept:
                current_concept["source_document"] = doc_path
                concepts.append(current_concept)
            
        except Exception as e:
            logger.error(f"Failed to parse concepts response: {str(e)}")
        
        return concepts
    
    async def _generate_factual_questions(self, facts: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate factual questions from extracted facts"""
        alpaca_examples = []
        
        for fact in facts[:20]:  # Limit to avoid too many examples
            try:
                # Create question about the fact
                fact_text = fact.get("fact", "")
                context = fact.get("context", "")
                
                # Generate question using LLM
                question_prompt = f"""
                Based on this fact: "{fact_text}"
                And this context: "{context}"
                
                Generate a clear, specific question that can be answered with the given information.
                The question should test understanding of the factual information.
                """
                
                model_spec = config.get("data_generation_model", "")
                question = await self.llm_manager.generate_response(model_spec, question_prompt, config)
                question = question.strip()
                
                # Generate answer using the fact and context
                answer_prompt = f"""
                Question: {question}
                
                Based on this information:
                Fact: {fact_text}
                Context: {context}
                
                Provide a clear, accurate answer to the question.
                """
                
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = answer.strip()
                
                # Create Alpaca format entry
                alpaca_example = {
                    "instruction": question,
                    "input": context if context else "",
                    "output": answer
                }
                
                alpaca_examples.append(alpaca_example)
                
            except Exception as e:
                logger.error(f"Failed to generate factual question: {str(e)}")
                continue
        
        return alpaca_examples
    
    async def _generate_analytical_questions(self, concepts: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate analytical questions from extracted concepts"""
        alpaca_examples = []
        
        # Generate comparison questions between concepts
        for i in range(min(10, len(concepts))):
            for j in range(i + 1, min(i + 3, len(concepts))):
                try:
                    concept1 = concepts[i]
                    concept2 = concepts[j]
                    
                    # Generate analytical question
                    question_prompt = f"""
                    Create an analytical question that compares or relates these two concepts:
                    Concept 1: {concept1.get('concept', '')}
                    Definition 1: {concept1.get('definition', '')}
                    
                    Concept 2: {concept2.get('concept', '')}
                    Definition 2: {concept2.get('definition', '')}
                    
                    Generate a question that requires analysis and comparison.
                    """
                    
                    model_spec = config.get("data_generation_model", "")
                    question = await self.llm_manager.generate_response(model_spec, question_prompt, config)
                    question = question.strip()
                    
                    # Generate analytical answer
                    answer_prompt = f"""
                    Question: {question}
                    
                    Analyze and compare these concepts:
                    Concept 1: {concept1.get('concept', '')} - {concept1.get('definition', '')}
                    Concept 2: {concept2.get('concept', '')} - {concept2.get('definition', '')}
                    
                    Provide a thoughtful analytical answer.
                    """
                    
                    answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                    answer = answer.strip()
                    
                    # Create Alpaca format entry
                    alpaca_example = {
                        "instruction": question,
                        "input": f"Concept 1: {concept1.get('concept', '')}\nConcept 2: {concept2.get('concept', '')}",
                        "output": answer
                    }
                    
                    alpaca_examples.append(alpaca_example)
                    
                except Exception as e:
                    logger.error(f"Failed to generate analytical question: {str(e)}")
                    continue
        
        return alpaca_examples
    
    async def _generate_application_questions(self, concepts: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate application-based questions from concepts"""
        alpaca_examples = []
        
        for concept in concepts[:15]:  # Limit to avoid too many examples
            try:
                # Generate application question
                question_prompt = f"""
                Based on this concept: {concept.get('concept', '')}
                Definition: {concept.get('definition', '')}
                
                Create a practical application question that asks how this concept can be used in real-world scenarios.
                """
                
                model_spec = config.get("data_generation_model", "")
                question = await self.llm_manager.generate_response(model_spec, question_prompt, config)
                question = question.strip()
                
                # Generate application answer
                answer_prompt = f"""
                Question: {question}
                
                Based on this concept: {concept.get('concept', '')}
                Definition: {concept.get('definition', '')}
                
                Provide practical examples and applications of this concept.
                """
                
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = answer.strip()
                
                # Create Alpaca format entry
                alpaca_example = {
                    "instruction": question,
                    "input": f"Concept: {concept.get('concept', '')}",
                    "output": answer
                }
                
                alpaca_examples.append(alpaca_example)
                
            except Exception as e:
                logger.error(f"Failed to generate application question: {str(e)}")
                continue
        
        return alpaca_examples
    
    async def _generate_rag_based_questions(self, document_paths: List[str], config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate questions that require RAG retrieval to answer"""
        alpaca_examples = []
        
        # Generate sample queries that would require document retrieval
        sample_queries = [
            "What are the main points discussed in the document?",
            "Summarize the key findings from the document.",
            "What evidence is provided to support the main arguments?",
            "What are the implications of the information presented?",
            "How do the different sections of the document relate to each other?"
        ]
        
        for query in sample_queries:
            try:
                # Use RAG to retrieve relevant information
                relevant_chunks = await self.rag_system.retrieve_relevant_chunks(query, top_k=3)
                
                if not relevant_chunks:
                    continue
                
                # Combine retrieved chunks as context
                context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
                
                # Generate answer based on retrieved context
                answer_prompt = f"""
                Question: {query}
                
                Based on the following information from the documents:
                {context}
                
                Provide a comprehensive answer to the question.
                """
                
                model_spec = config.get("data_generation_model", "")
                answer = await self.llm_manager.generate_response(model_spec, answer_prompt, config)
                answer = answer.strip()
                
                # Create Alpaca format entry
                alpaca_example = {
                    "instruction": query,
                    "input": "Based on the provided documents",
                    "output": answer
                }
                
                alpaca_examples.append(alpaca_example)
                
            except Exception as e:
                logger.error(f"Failed to generate RAG-based question: {str(e)}")
                continue
        
        return alpaca_examples
    
    def _validate_alpaca_format(self, alpaca_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate and clean Alpaca format data"""
        validated_data = []
        
        for example in alpaca_data:
            try:
                # Check required fields
                if not all(key in example for key in ["instruction", "input", "output"]):
                    continue
                
                # Clean and validate content
                instruction = example["instruction"].strip()
                input_text = example["input"].strip()
                output_text = example["output"].strip()
                
                # Skip if any field is empty or too short
                if len(instruction) < 10 or len(output_text) < 10:
                    continue
                
                # Skip if output is just a repetition of instruction
                if instruction.lower() in output_text.lower() and len(output_text) < len(instruction) * 1.5:
                    continue
                
                validated_example = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                }
                
                validated_data.append(validated_example)
                
            except Exception as e:
                logger.error(f"Failed to validate example: {str(e)}")
                continue
        
        return validated_data
    
    def save_alpaca_dataset(self, alpaca_data: List[Dict[str, str]], output_path: str) -> bool:
        """Save Alpaca dataset to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Alpaca dataset saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Alpaca dataset: {str(e)}")
            return False
