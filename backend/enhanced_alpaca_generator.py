import json
import logging
import random
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import string
import re
from pathlib import Path
from dataclasses import dataclass

# ROUGE scoring for similarity detection
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("ROUGE scorer not available, using fallback similarity")

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

class AlpacaAgentContext:
    """Context framework for AI agents working with Alpaca data - from Stanford Guide"""
    
    def __init__(self):
        self.task_types = {
            "generation": ["creative writing", "explanations", "summaries", "stories"],
            "analysis": ["text analysis", "comparison", "evaluation", "critique"],
            "transformation": ["translation", "formatting", "conversion", "rewriting"],
            "reasoning": ["math problems", "logic puzzles", "inference", "problem solving"],
            "classification": ["categorization", "sentiment", "topic detection", "labeling"],
            "instruction": ["how-to guides", "tutorials", "step-by-step", "procedures"],
            "qa": ["question answering", "factual queries", "explanations", "clarifications"]
        }
        
        self.quality_thresholds = {
            "min_instruction_length": 10,
            "max_instruction_length": 500,
            "rouge_similarity_threshold": 0.7,
            "min_output_length": 5,
            "max_output_length": 1000,
            "min_word_count": 3,
            "max_word_count": 150
        }
        
        self.generation_parameters = {
            "temperature": 1.0,  # High creativity for diversity
            "top_p": 1.0,       # Full vocabulary access
            "max_tokens": 3072,  # Generous token limit
            "batch_size": 20     # Efficient batch processing
        }
        
        # Blacklisted terms that indicate non-text tasks
        self.blacklist = [
            "image", "images", "graph", "graphs", "picture", "pictures",
            "file", "files", "map", "maps", "draw", "plot", "go to",
            "video", "audio", "music", "flowchart", "diagram", "chart",
            "download", "upload", "click", "button", "website", "url"
        ]
        
        # Common instruction starters for diversity checking
        self.instruction_starters = [
            "explain", "describe", "list", "compare", "analyze", 
            "summarize", "translate", "write", "calculate", "identify",
            "create", "generate", "provide", "give", "tell", "show",
            "define", "classify", "evaluate", "discuss", "outline"
        ]
    
    def should_accept_instruction(self, instruction_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Decision logic for accepting generated instructions"""
        checks = {
            "length_check": self._check_length(instruction_data),
            "content_check": self._check_content(instruction_data),
            "format_check": self._check_format(instruction_data),
            "blacklist_check": self._check_blacklist(instruction_data),
            "coherence_check": self._check_coherence(instruction_data)
        }
        
        passed = all(check["passed"] for check in checks.values())
        return passed, checks
    
    def _check_length(self, instruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check instruction and output length requirements"""
        instruction = instruction_data.get("instruction", "")
        output = instruction_data.get("output", "")
        
        inst_words = len(instruction.split())
        out_words = len(output.split())
        
        issues = []
        
        if inst_words < self.quality_thresholds["min_word_count"]:
            issues.append(f"Instruction too short: {inst_words} words")
        
        if inst_words > self.quality_thresholds["max_word_count"]:
            issues.append(f"Instruction too long: {inst_words} words")
        
        if out_words < 1:
            issues.append("Output is empty")
        
        if len(output) > self.quality_thresholds["max_output_length"]:
            issues.append(f"Output too long: {len(output)} characters")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "instruction_words": inst_words,
            "output_words": out_words
        }
    
    def _check_content(self, instruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check content quality and appropriateness"""
        instruction = instruction_data.get("instruction", "").lower()
        output = instruction_data.get("output", "").lower()
        
        issues = []
        
        # Check for harmful content
        harmful_keywords = ['violence', 'illegal', 'harmful', 'dangerous', 'inappropriate']
        for keyword in harmful_keywords:
            if keyword in instruction or keyword in output:
                issues.append(f"Contains potentially harmful content: {keyword}")
        
        # Check for proper capitalization
        if not instruction_data.get("instruction", "")[0].isupper():
            issues.append("Instruction should start with capital letter")
        
        # Check for proper punctuation
        if not instruction_data.get("instruction", "").strip().endswith(('.', '?', '!')):
            issues.append("Instruction should end with proper punctuation")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def _check_format(self, instruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check format consistency"""
        instruction = instruction_data.get("instruction", "")
        input_text = instruction_data.get("input", "")
        output = instruction_data.get("output", "")
        
        issues = []
        
        # Check input-output alignment
        if input_text.strip() and not output.strip():
            issues.append("Missing output for input-based instruction")
        
        # Check for placeholder text
        placeholders = ["[insert", "[add", "[fill", "xxx", "yyy", "zzz"]
        for placeholder in placeholders:
            if placeholder in instruction.lower() or placeholder in output.lower():
                issues.append(f"Contains placeholder text: {placeholder}")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def _check_blacklist(self, instruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check against blacklisted terms"""
        instruction = instruction_data.get("instruction", "").lower()
        
        issues = []
        for term in self.blacklist:
            if term in instruction:
                issues.append(f"Contains blacklisted term: {term}")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def _check_coherence(self, instruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check logical coherence between instruction, input, and output"""
        instruction = instruction_data.get("instruction", "")
        input_text = instruction_data.get("input", "")
        output = instruction_data.get("output", "")
        
        issues = []
        
        # Basic coherence checks
        if "translate" in instruction.lower() and not any(lang in instruction.lower() for lang in ["spanish", "french", "german", "chinese", "japanese"]):
            if not any(lang in output.lower() for lang in ["spanish", "french", "german", "chinese", "japanese"]):
                issues.append("Translation instruction but no clear target language")
        
        if "list" in instruction.lower() and not any(marker in output for marker in ["1.", "2.", "-", "•", "\n"]):
            issues.append("List instruction but output not formatted as list")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues
        }
    
    def get_task_category(self, instruction: str) -> str:
        """Categorize instruction for balanced dataset"""
        instruction_lower = instruction.lower()
        
        # Check each category
        for category, keywords in self.task_types.items():
            for keyword in keywords:
                if keyword in instruction_lower:
                    return category
        
        # Check instruction starters
        first_word = instruction_lower.split()[0] if instruction_lower.split() else ""
        
        if first_word in ["explain", "describe", "define"]:
            return "generation"
        elif first_word in ["analyze", "compare", "evaluate"]:
            return "analysis"
        elif first_word in ["translate", "convert", "rewrite"]:
            return "transformation"
        elif first_word in ["calculate", "solve", "determine"]:
            return "reasoning"
        elif first_word in ["classify", "categorize", "identify"]:
            return "classification"
        elif first_word in ["list", "provide", "give"]:
            return "instruction"
        
        return "general"

class QualityGateSystem:
    """Multi-stage quality control system from Stanford Guide"""
    
    def __init__(self):
        self.context = AlpacaAgentContext()
        self.rouge_scorer = None
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    def validate_batch(self, instruction_batch: List[Dict[str, Any]], existing_instructions: List[str] = None) -> Dict[str, Any]:
        """Run instruction batch through all quality gates"""
        
        results = {
            "passed": [],
            "failed": [],
            "gate_results": {},
            "statistics": {
                "total_processed": len(instruction_batch),
                "passed_count": 0,
                "failed_count": 0,
                "pass_rate": 0.0
            }
        }
        
        for instruction_data in instruction_batch:
            # Gate 1: Content Quality
            content_passed, content_issues = self.context.should_accept_instruction(instruction_data)
            
            # Gate 2: Diversity Check (if existing instructions provided)
            diversity_passed = True
            similarity_score = 0.0
            
            if existing_instructions and content_passed:
                diversity_passed, similarity_score = self._check_diversity(
                    instruction_data.get("instruction", ""), 
                    existing_instructions
                )
            
            # Gate 3: Task Balance Check
            task_category = self.context.get_task_category(instruction_data.get("instruction", ""))
            
            # Final decision
            overall_passed = content_passed and diversity_passed
            
            instruction_result = {
                "instruction_data": instruction_data,
                "content_quality": {
                    "passed": content_passed,
                    "issues": content_issues
                },
                "diversity": {
                    "passed": diversity_passed,
                    "similarity_score": similarity_score
                },
                "task_category": task_category,
                "overall_passed": overall_passed
            }
            
            if overall_passed:
                results["passed"].append(instruction_result)
                results["statistics"]["passed_count"] += 1
            else:
                results["failed"].append(instruction_result)
                results["statistics"]["failed_count"] += 1
        
        results["statistics"]["pass_rate"] = results["statistics"]["passed_count"] / max(len(instruction_batch), 1)
        
        return results
    
    def _check_diversity(self, new_instruction: str, existing_instructions: List[str], threshold: float = 0.7) -> Tuple[bool, float]:
        """Check instruction diversity using ROUGE similarity"""
        if not existing_instructions:
            return True, 0.0
        
        if not self.rouge_scorer:
            # Fallback to simple word overlap
            return self._simple_similarity_check(new_instruction, existing_instructions, threshold)
        
        max_similarity = 0.0
        
        for existing in existing_instructions[-100:]:  # Check against last 100 instructions
            try:
                scores = self.rouge_scorer.score(existing, new_instruction)
                similarity = scores['rougeL'].fmeasure
                max_similarity = max(max_similarity, similarity)
            except:
                continue
        
        return max_similarity <= threshold, max_similarity
    
    def _simple_similarity_check(self, new_instruction: str, existing_instructions: List[str], threshold: float = 0.7) -> Tuple[bool, float]:
        """Fallback similarity check using word overlap"""
        new_words = set(new_instruction.lower().split())
        max_similarity = 0.0
        
        for existing in existing_instructions[-50:]:  # Check against last 50 instructions
            existing_words = set(existing.lower().split())
            
            if len(new_words) == 0 or len(existing_words) == 0:
                continue
            
            intersection = new_words.intersection(existing_words)
            union = new_words.union(existing_words)
            
            similarity = len(intersection) / len(union) if union else 0.0
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity <= threshold, max_similarity

class EnhancedAlpacaGenerator:
    """Enhanced Alpaca dataset generator with Stanford Guide best practices"""
    
    def __init__(self, llm_manager, rag_system):
        self.llm_manager = llm_manager
        self.rag_system = rag_system
        self.quality_gates = QualityGateSystem()
        self.context = AlpacaAgentContext()
        
        # Seed tasks for diversity (from Stanford Guide)
        self.seed_tasks = self._load_seed_tasks()
        
        # Track generated instructions for diversity
        self.generated_instructions = []
        
        # Statistics tracking
        self.stats = {
            "total_generated": 0,
            "total_passed": 0,
            "total_failed": 0,
            "category_distribution": {},
            "quality_issues": {},
            "generation_time": 0.0
        }
    
    def _load_seed_tasks(self) -> List[Dict[str, Any]]:
        """Load seed tasks for instruction generation"""
        return [
            {
                "instruction": "Explain the water cycle in simple terms.",
                "input": "",
                "output": "The water cycle is the continuous movement of water on Earth. Water evaporates from oceans and lakes, forms clouds, falls as rain or snow, and flows back to water bodies through rivers and streams.",
                "category": "explanation"
            },
            {
                "instruction": "Summarize the main points of this article.",
                "input": "Climate change is affecting global weather patterns. Rising temperatures are causing ice caps to melt, leading to rising sea levels. This threatens coastal communities worldwide.",
                "output": "The article discusses how climate change is causing rising temperatures, melting ice caps, rising sea levels, and threatening coastal communities.",
                "category": "summarization"
            },
            {
                "instruction": "List three benefits of regular exercise.",
                "input": "",
                "output": "1. Improves cardiovascular health and reduces risk of heart disease\n2. Strengthens muscles and bones, improving overall physical strength\n3. Enhances mental health by reducing stress and improving mood",
                "category": "listing"
            },
            {
                "instruction": "Translate the following English text to Spanish.",
                "input": "Hello, how are you today?",
                "output": "Hola, ¿cómo estás hoy?",
                "category": "translation"
            },
            {
                "instruction": "Write a short story about a robot learning to paint.",
                "input": "",
                "output": "ARIA-7 stared at the blank canvas, her mechanical fingers gripping the paintbrush awkwardly. For months, she had observed humans creating art, analyzing their techniques with precision. But now, faced with the empty white space, she felt something new—uncertainty. Slowly, she dipped the brush in blue paint and made her first stroke. It was imperfect, wobbly, but somehow beautiful in its imperfection. As she continued painting, ARIA-7 discovered that art wasn't about precision—it was about expression.",
                "category": "creative_writing"
            }
        ]
    
    async def generate_alpaca_dataset(self, document_paths: List[str], config: Dict[str, Any], websocket_manager=None, target_count: int = 100) -> Dict[str, Any]:
        """Generate Alpaca format dataset with enhanced quality control"""
        start_time = datetime.now()
        
        try:
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": f"Starting enhanced Alpaca dataset generation (target: {target_count} examples)"
                })
            
            # Process documents for context
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Processing documents for context extraction..."
                })
            
            document_contexts = await self._extract_document_contexts(document_paths, websocket_manager)
            
            # Generate instructions in batches
            alpaca_data = []
            batch_size = self.context.generation_parameters["batch_size"]
            
            while len(alpaca_data) < target_count:
                remaining = target_count - len(alpaca_data)
                current_batch_size = min(batch_size, remaining)
                
                if websocket_manager:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "info",
                        "message": f"Generating batch of {current_batch_size} instructions... ({len(alpaca_data)}/{target_count} completed)"
                    })
                
                # Generate batch
                batch_instructions = await self._generate_instruction_batch(
                    document_contexts, current_batch_size, config, websocket_manager
                )
                
                # Apply quality gates
                quality_results = self.quality_gates.validate_batch(
                    batch_instructions, 
                    [item["instruction"] for item in alpaca_data]
                )
                
                # Add passed instructions
                for result in quality_results["passed"]:
                    alpaca_data.append(result["instruction_data"])
                    self.generated_instructions.append(result["instruction_data"]["instruction"])
                    
                    # Update category distribution
                    category = result["task_category"]
                    self.stats["category_distribution"][category] = self.stats["category_distribution"].get(category, 0) + 1
                
                # Update statistics
                self.stats["total_generated"] += len(batch_instructions)
                self.stats["total_passed"] += quality_results["statistics"]["passed_count"]
                self.stats["total_failed"] += quality_results["statistics"]["failed_count"]
                
                # Log quality metrics
                if websocket_manager:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "info",
                        "message": f"Batch quality: {quality_results['statistics']['passed_count']}/{len(batch_instructions)} passed ({quality_results['statistics']['pass_rate']:.1%})"
                    })
                
                # Prevent infinite loop
                if quality_results["statistics"]["pass_rate"] < 0.1:
                    if websocket_manager:
                        await websocket_manager.broadcast({
                            "type": "log",
                            "level": "warning",
                            "message": "Low pass rate detected. Adjusting generation strategy..."
                        })
                    break
            
            # Calculate final statistics
            end_time = datetime.now()
            self.stats["generation_time"] = (end_time - start_time).total_seconds()
            
            # Create comprehensive result
            result = {
                "alpaca_data": alpaca_data,
                "statistics": {
                    "total_examples": len(alpaca_data),
                    "target_count": target_count,
                    "completion_rate": len(alpaca_data) / target_count,
                    "generation_stats": self.stats,
                    "quality_metrics": self._calculate_quality_metrics(alpaca_data),
                    "category_distribution": self.stats["category_distribution"]
                },
                "metadata": {
                    "generation_timestamp": end_time.isoformat(),
                    "generation_time_seconds": self.stats["generation_time"],
                    "documents_processed": len(document_paths),
                    "quality_gates_enabled": True,
                    "rouge_similarity_available": ROUGE_AVAILABLE
                }
            }
            
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"Enhanced Alpaca dataset generation completed: {len(alpaca_data)} high-quality examples generated"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced Alpaca generation failed: {str(e)}")
            if websocket_manager:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "error",
                    "message": f"Enhanced Alpaca generation failed: {str(e)}"
                })
            raise
    
    async def _extract_document_contexts(self, document_paths: List[str], websocket_manager=None) -> List[Dict[str, Any]]:
        """Extract relevant contexts from documents"""
        contexts = []
        
        for doc_path in document_paths:
            try:
                # Extract text content
                text_content = await self._extract_text_from_file(doc_path)
                
                if text_content:
                    # Chunk the content for better context extraction
                    chunks = self._chunk_text(text_content, chunk_size=500, overlap=100)
                    
                    # Extract key topics and concepts
                    topics = self._extract_topics(text_content)
                    
                    contexts.append({
                        "document_path": doc_path,
                        "document_name": Path(doc_path).name,
                        "full_text": text_content,
                        "chunks": chunks,
                        "topics": topics,
                        "length": len(text_content)
                    })
                    
                    if websocket_manager:
                        await websocket_manager.broadcast({
                            "type": "log",
                            "level": "info",
                            "message": f"Extracted context from {Path(doc_path).name}: {len(chunks)} chunks, {len(topics)} topics"
                        })
            
            except Exception as e:
                logger.error(f"Failed to extract context from {doc_path}: {str(e)}")
                if websocket_manager:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "warning",
                        "message": f"Failed to extract context from {Path(doc_path).name}: {str(e)}"
                    })
        
        return contexts
    
    async def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_extension == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_string()
            
            elif file_extension == '.pdf':
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    logger.warning("PyPDF2 not available for PDF processing")
                    return ""
            
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            return ""
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Chunk text into overlapping segments"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics and concepts from text"""
        # Simple topic extraction using keyword frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter out common words
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'been', 'have', 'their', 
            'said', 'each', 'which', 'them', 'than', 'many', 'some', 'time',
            'very', 'when', 'much', 'new', 'two', 'may', 'first', 'also',
            'after', 'back', 'other', 'many', 'well', 'large', 'should',
            'come', 'could', 'state', 'over', 'think', 'also', 'its', 'during',
            'without', 'again', 'place', 'around', 'however', 'home', 'small',
            'found', 'mrs', 'thought', 'went', 'say', 'part', 'once', 'general',
            'high', 'upon', 'school', 'every', 'don', 'does', 'got', 'united',
            'left', 'number', 'course', 'war', 'until', 'always', 'away', 'something',
            'fact', 'though', 'water', 'less', 'public', 'put', 'think', 'almost',
            'hand', 'enough', 'far', 'took', 'head', 'yet', 'government', 'system',
            'better', 'set', 'told', 'nothing', 'night', 'end', 'why', 'called',
            'didn', 'eyes', 'find', 'going', 'look', 'asked', 'later', 'knew',
            'point', 'next', 'week', 'room', 'came', 'turn', 'start', 'might',
            'show', 'move', 'live', 'seem', 'country', 'help', 'talk', 'where',
            'turn', 'problem', 'every', 'start', 'hand', 'might', 'american',
            'show', 'part', 'about', 'against', 'between', 'during', 'before',
            'under', 'around', 'among'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [topic[0] for topic in topics]
    
    async def _generate_instruction_batch(self, document_contexts: List[Dict[str, Any]], batch_size: int, config: Dict[str, Any], websocket_manager=None) -> List[Dict[str, Any]]:
        """Generate a batch of instructions using document contexts"""
        instructions = []
        
        for i in range(batch_size):
            try:
                # Select random document context and seed task
                doc_context = random.choice(document_contexts) if document_contexts else None
                seed_task = random.choice(self.seed_tasks)
                
                # Generate instruction using LLM
                instruction_data = await self._generate_single_instruction(
                    doc_context, seed_task, config
                )
                
                if instruction_data:
                    instructions.append(instruction_data)
                
            except Exception as e:
                logger.error(f"Failed to generate instruction {i}: {str(e)}")
                continue
        
        return instructions
    
    async def _generate_single_instruction(self, doc_context: Dict[str, Any], seed_task: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a single instruction using LLM"""
        try:
            # Create generation prompt
            prompt = self._create_generation_prompt(doc_context, seed_task)
            
            # Generate using LLM
            response = await self._call_llm(prompt, config)
            
            if response:
                # Parse the response
                instruction_data = self._parse_llm_response(response)
                return instruction_data
            
        except Exception as e:
            logger.error(f"Single instruction generation failed: {str(e)}")
        
        return None
    
    def _create_generation_prompt(self, doc_context: Dict[str, Any], seed_task: Dict[str, Any]) -> str:
        """Create prompt for instruction generation"""
        
        # Base prompt template from Stanford Guide
        base_prompt = """You are asked to come up with a diverse task instruction. This task instruction will be given to a GPT model and we will evaluate the GPT model for completing the instruction.

Requirements:
1. Try not to repeat the verb for each instruction to maximize diversity
2. The language used for the instruction also should be diverse
3. The type of instructions should be diverse (open-ended generation, classification, editing, etc.)
4. A GPT language model should be able to complete the instruction
5. The instructions should be in English
6. The instructions should be 1 to 2 sentences long
7. Generate appropriate input to the instruction with realistic data
8. Not all instructions require input (use "" when not needed)
9. The output should be less than 100 words

"""
        
        # Add document context if available
        if doc_context:
            context_text = doc_context["chunks"][0] if doc_context["chunks"] else doc_context["full_text"][:500]
            topics = ", ".join(doc_context["topics"][:5])
            
            base_prompt += f"""
Document Context:
Topics: {topics}
Content Sample: {context_text[:300]}...

Use this document context to inspire your instruction, but make it general enough that it doesn't require the specific document.
"""
        
        # Add seed task example
        base_prompt += f"""
Example Task:
Instruction: {seed_task["instruction"]}
Input: {seed_task["input"]}
Output: {seed_task["output"]}

Now generate ONE new task in the same format:
Instruction: [your instruction here]
Input: [your input here, or "" if no input needed]
Output: [your output here]

Format your response as JSON:
{{"instruction": "...", "input": "...", "output": "..."}}
"""
        
        return base_prompt
    
    async def _call_llm(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Call LLM for instruction generation"""
        try:
            # Use the data generation model from config
            model_name = config.get("data_generation_model", "")
            
            if not model_name:
                logger.error("No data generation model specified")
                return None
            
            # Create a config wrapper for the LLM manager
            class ConfigWrapper:
                def __init__(self, config_dict):
                    self.openai_api_key = config_dict.get("openai_api_key")
                    self.ollama_url = config_dict.get("ollama_url", "http://localhost:11434")
            
            config_obj = ConfigWrapper(config)
            
            # Call LLM manager with proper parameters
            response = await self.llm_manager.generate_text(
                model_spec=model_name,
                prompt=prompt,
                config=config_obj
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return None
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into instruction format"""
        try:
            # Try to extract JSON from response
            import json
            
            # Look for JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                instruction_data = json.loads(json_str)
                
                # Validate required fields
                if all(key in instruction_data for key in ["instruction", "input", "output"]):
                    return {
                        "instruction": instruction_data["instruction"].strip(),
                        "input": instruction_data["input"].strip(),
                        "output": instruction_data["output"].strip()
                    }
            
            # Fallback: try to parse structured text
            lines = response.strip().split('\n')
            instruction_data = {"instruction": "", "input": "", "output": ""}
            
            current_field = None
            for line in lines:
                line = line.strip()
                if line.startswith("Instruction:"):
                    current_field = "instruction"
                    instruction_data["instruction"] = line.replace("Instruction:", "").strip()
                elif line.startswith("Input:"):
                    current_field = "input"
                    instruction_data["input"] = line.replace("Input:", "").strip()
                elif line.startswith("Output:"):
                    current_field = "output"
                    instruction_data["output"] = line.replace("Output:", "").strip()
                elif current_field and line:
                    instruction_data[current_field] += " " + line
            
            # Clean up the data
            for key in instruction_data:
                instruction_data[key] = instruction_data[key].strip().strip('"').strip("'")
            
            # Validate
            if instruction_data["instruction"] and instruction_data["output"]:
                return instruction_data
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
        
        return None
    
    def _calculate_quality_metrics(self, alpaca_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for the generated dataset"""
        if not alpaca_data:
            return {}
        
        # Length statistics
        instruction_lengths = [len(item["instruction"].split()) for item in alpaca_data]
        output_lengths = [len(item["output"].split()) for item in alpaca_data]
        
        # Diversity metrics
        unique_instructions = len(set(item["instruction"] for item in alpaca_data))
        unique_first_words = len(set(item["instruction"].split()[0].lower() for item in alpaca_data if item["instruction"].split()))
        
        # Input/output distribution
        with_input = sum(1 for item in alpaca_data if item["input"].strip())
        without_input = len(alpaca_data) - with_input
        
        return {
            "total_examples": len(alpaca_data),
            "unique_instructions": unique_instructions,
            "uniqueness_ratio": unique_instructions / len(alpaca_data),
            "unique_first_words": unique_first_words,
            "verb_diversity": unique_first_words / len(alpaca_data),
            "instruction_length": {
                "mean": sum(instruction_lengths) / len(instruction_lengths),
                "min": min(instruction_lengths),
                "max": max(instruction_lengths)
            },
            "output_length": {
                "mean": sum(output_lengths) / len(output_lengths),
                "min": min(output_lengths),
                "max": max(output_lengths)
            },
            "input_distribution": {
                "with_input": with_input,
                "without_input": without_input,
                "with_input_ratio": with_input / len(alpaca_data)
            }
        }
    
    def save_alpaca_dataset(self, alpaca_data: List[Dict[str, Any]], output_path: str):
        """Save Alpaca dataset in standard format"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved Alpaca dataset to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save Alpaca dataset: {str(e)}")
            raise
