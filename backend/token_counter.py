import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import tiktoken

# Document processing imports
try:
    import PyPDF2
    import pandas as pd
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PDF processing not available")

logger = logging.getLogger(__name__)

class TokenCounter:
    """Token counting utility for documents and text content"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter with specified encoding
        cl100k_base is used by GPT-4, GPT-3.5-turbo, and text-embedding-ada-002
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.encoding_name = encoding_name
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
            # Fallback to simple word-based counting
            self.encoding = None
            self.encoding_name = "word_based_fallback"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        try:
            if self.encoding:
                return len(self.encoding.encode(text))
            else:
                # Fallback: approximate tokens as words * 1.3 (rough estimate)
                words = len(text.split())
                return int(words * 1.3)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            # Ultra-fallback: character count / 4 (very rough estimate)
            return len(text) // 4
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
            
            elif file_extension == '.pdf' and PDF_AVAILABLE:
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            return ""
    
    def count_tokens_in_file(self, file_path: str) -> Dict[str, Any]:
        """Count tokens in a file and return detailed information"""
        try:
            if not os.path.exists(file_path):
                return {
                    "error": "File not found",
                    "token_count": 0,
                    "character_count": 0,
                    "word_count": 0
                }
            
            # Extract text content
            text_content = self.extract_text_from_file(file_path)
            
            if not text_content:
                return {
                    "error": "No text content extracted",
                    "token_count": 0,
                    "character_count": 0,
                    "word_count": 0
                }
            
            # Count tokens, characters, and words
            token_count = self.count_tokens(text_content)
            character_count = len(text_content)
            word_count = len(text_content.split())
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            return {
                "token_count": token_count,
                "character_count": character_count,
                "word_count": word_count,
                "file_size": file_size,
                "encoding": self.encoding_name,
                "text_length": len(text_content)
            }
            
        except Exception as e:
            logger.error(f"Token counting failed for {file_path}: {str(e)}")
            return {
                "error": str(e),
                "token_count": 0,
                "character_count": 0,
                "word_count": 0
            }
    
    def count_tokens_in_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """Count tokens across multiple documents"""
        try:
            document_stats = []
            total_tokens = 0
            total_characters = 0
            total_words = 0
            total_files = 0
            successful_files = 0
            
            for doc_path in document_paths:
                doc_name = os.path.basename(doc_path)
                stats = self.count_tokens_in_file(doc_path)
                
                # Add document name to stats
                stats["document_name"] = doc_name
                stats["document_path"] = doc_path
                
                document_stats.append(stats)
                total_files += 1
                
                if "error" not in stats:
                    total_tokens += stats["token_count"]
                    total_characters += stats["character_count"]
                    total_words += stats["word_count"]
                    successful_files += 1
            
            return {
                "documents": document_stats,
                "summary": {
                    "total_tokens": total_tokens,
                    "total_characters": total_characters,
                    "total_words": total_words,
                    "total_files": total_files,
                    "successful_files": successful_files,
                    "failed_files": total_files - successful_files,
                    "encoding": self.encoding_name
                }
            }
            
        except Exception as e:
            logger.error(f"Batch token counting failed: {str(e)}")
            return {
                "error": str(e),
                "documents": [],
                "summary": {
                    "total_tokens": 0,
                    "total_characters": 0,
                    "total_words": 0,
                    "total_files": 0,
                    "successful_files": 0,
                    "failed_files": 0,
                    "encoding": self.encoding_name
                }
            }
    
    def estimate_context_window_usage(self, token_count: int, context_window_size: int = 4096) -> Dict[str, Any]:
        """Estimate how much of a context window the tokens would use"""
        try:
            usage_percentage = (token_count / context_window_size) * 100
            remaining_tokens = max(0, context_window_size - token_count)
            
            # Determine status
            if usage_percentage < 50:
                status = "low"
            elif usage_percentage < 80:
                status = "medium"
            elif usage_percentage < 95:
                status = "high"
            else:
                status = "critical"
            
            return {
                "token_count": token_count,
                "context_window_size": context_window_size,
                "usage_percentage": round(usage_percentage, 2),
                "remaining_tokens": remaining_tokens,
                "status": status,
                "fits_in_context": token_count <= context_window_size
            }
            
        except Exception as e:
            logger.error(f"Context window estimation failed: {str(e)}")
            return {
                "error": str(e),
                "token_count": token_count,
                "context_window_size": context_window_size,
                "usage_percentage": 0,
                "remaining_tokens": context_window_size,
                "status": "unknown",
                "fits_in_context": True
            }
    
    def get_model_context_windows(self) -> Dict[str, int]:
        """Get common model context window sizes"""
        return {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "claude-instant": 100000,
            "claude-2": 100000,
            "llama-2-7b": 4096,
            "llama-2-13b": 4096,
            "llama-2-70b": 4096,
            "mistral-7b": 8192,
            "mixtral-8x7b": 32768,
            "gemma-7b": 8192,
            "phi-3": 4096,
            "qwen-7b": 8192,
            "codellama": 16384
        }
    
    def format_token_count(self, token_count: int) -> str:
        """Format token count for display"""
        if token_count < 1000:
            return f"{token_count} tokens"
        elif token_count < 1000000:
            return f"{token_count/1000:.1f}K tokens"
        else:
            return f"{token_count/1000000:.1f}M tokens"
