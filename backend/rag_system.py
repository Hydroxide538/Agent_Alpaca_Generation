import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import numpy as np
from pathlib import Path

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available, falling back to simple similarity")

# Document processing imports
try:
    import PyPDF2
    import pandas as pd
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PDF processing not available")

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available")

logger = logging.getLogger(__name__)

class RAGSystem:
    """Enhanced RAG system with proper embedding and retrieval capabilities"""
    
    def __init__(self, embedding_model: str = None, reranking_model: str = None, ollama_url: str = "http://localhost:11434"):
        self.embedding_model = embedding_model
        self.reranking_model = reranking_model
        self.ollama_url = ollama_url
        self.vector_db = None
        self.collection = None
        self.embeddings_cache = {}
        
        # Initialize vector database
        self._initialize_vector_db()
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        try:
            if CHROMADB_AVAILABLE:
                # Create persistent ChromaDB client
                db_path = os.path.join(os.getcwd(), "vector_db")
                os.makedirs(db_path, exist_ok=True)
                
                self.vector_db = chromadb.PersistentClient(
                    path=db_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Create or get collection
                collection_name = "document_embeddings"
                try:
                    self.collection = self.vector_db.get_collection(collection_name)
                    logger.info(f"Using existing ChromaDB collection: {collection_name}")
                except:
                    self.collection = self.vector_db.create_collection(
                        name=collection_name,
                        metadata={"description": "Document embeddings for RAG system"}
                    )
                    logger.info(f"Created new ChromaDB collection: {collection_name}")
            else:
                logger.warning("ChromaDB not available, using in-memory storage")
                self.embeddings_cache = {}
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            self.vector_db = None
            self.collection = None
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        try:
            if self.embedding_model and "ollama:" in self.embedding_model:
                # Use Ollama embedding model
                self.embedding_client = "ollama"
                logger.info(f"Using Ollama embedding model: {self.embedding_model}")
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                # Fallback to sentence transformers
                model_name = "all-MiniLM-L6-v2"  # Lightweight but effective
                self.embedding_client = SentenceTransformer(model_name)
                logger.info(f"Using SentenceTransformer model: {model_name}")
            else:
                logger.warning("No embedding model available, using simple text similarity")
                self.embedding_client = None
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.embedding_client = None
    
    async def process_documents(self, document_paths: List[str], websocket_manager=None) -> Dict[str, Any]:
        """Process documents and create embeddings"""
        try:
            processed_docs = []
            total_chunks = 0
            
            for doc_path in document_paths:
                if websocket_manager:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "info",
                        "message": f"Processing document: {os.path.basename(doc_path)}"
                    })
                
                # Extract text from document
                text_content = await self._extract_text(doc_path)
                
                if not text_content:
                    logger.warning(f"No text extracted from {doc_path}")
                    continue
                
                # Chunk the document
                chunks = self._chunk_text(text_content)
                
                # Create embeddings for chunks
                embeddings = await self._create_embeddings(chunks)
                
                # Store in vector database
                doc_id = os.path.basename(doc_path)
                await self._store_embeddings(doc_id, chunks, embeddings)
                
                processed_docs.append({
                    "document_id": doc_id,
                    "path": doc_path,
                    "chunks_count": len(chunks),
                    "text_length": len(text_content)
                })
                
                total_chunks += len(chunks)
                
                if websocket_manager:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "success",
                        "message": f"Processed {doc_id}: {len(chunks)} chunks created"
                    })
            
            return {
                "processed_documents": processed_docs,
                "total_chunks": total_chunks,
                "embedding_model": self.embedding_model,
                "vector_db_status": "active" if self.collection else "fallback"
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
    
    async def _extract_text(self, file_path: str) -> str:
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
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk text into overlapping segments"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
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
    
    async def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        try:
            if self.embedding_client == "ollama":
                return await self._create_ollama_embeddings(texts)
            elif isinstance(self.embedding_client, SentenceTransformer):
                embeddings = self.embedding_client.encode(texts)
                return embeddings.tolist()
            else:
                # Fallback: simple hash-based embeddings
                return [[hash(text) % 1000 / 1000.0] * 384 for text in texts]
        except Exception as e:
            logger.error(f"Embedding creation failed: {str(e)}")
            # Return dummy embeddings
            return [[0.0] * 384 for _ in texts]
    
    async def _create_ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using Ollama"""
        try:
            import aiohttp
            
            model_name = self.embedding_model.replace("ollama:", "")
            embeddings = []
            
            async with aiohttp.ClientSession() as session:
                for text in texts:
                    payload = {
                        "model": model_name,
                        "prompt": text
                    }
                    
                    async with session.post(
                        f"{self.ollama_url}/api/embeddings",
                        json=payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            embeddings.append(result.get("embedding", [0.0] * 384))
                        else:
                            logger.warning(f"Ollama embedding failed for text chunk")
                            embeddings.append([0.0] * 384)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Ollama embedding creation failed: {str(e)}")
            return [[0.0] * 384 for _ in texts]
    
    async def _store_embeddings(self, doc_id: str, chunks: List[str], embeddings: List[List[float]]):
        """Store embeddings in vector database"""
        try:
            if self.collection:
                # Store in ChromaDB
                ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
                metadatas = [{"document_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
                
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                # Store in memory cache
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = f"{doc_id}_{i}"
                    self.embeddings_cache[chunk_id] = {
                        "text": chunk,
                        "embedding": embedding,
                        "document_id": doc_id,
                        "chunk_index": i
                    }
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
    
    async def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        try:
            # Create query embedding
            query_embeddings = await self._create_embeddings([query])
            query_embedding = query_embeddings[0]
            
            if self.collection:
                # Query ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                relevant_chunks = []
                for i in range(len(results['documents'][0])):
                    relevant_chunks.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else 0.0
                    })
                
                return relevant_chunks
            else:
                # Use in-memory cache with simple similarity
                similarities = []
                for chunk_id, chunk_data in self.embeddings_cache.items():
                    similarity = self._cosine_similarity(query_embedding, chunk_data["embedding"])
                    similarities.append((chunk_id, similarity, chunk_data))
                
                # Sort by similarity and return top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                relevant_chunks = []
                for chunk_id, similarity, chunk_data in similarities[:top_k]:
                    relevant_chunks.append({
                        "text": chunk_data["text"],
                        "metadata": {
                            "document_id": chunk_data["document_id"],
                            "chunk_index": chunk_data["chunk_index"]
                        },
                        "similarity": similarity
                    })
                
                return relevant_chunks
                
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    async def rerank_results(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using reranking model (if available)"""
        try:
            if not self.reranking_model or not chunks:
                return chunks
            
            # For now, use simple text similarity as reranking
            # In a full implementation, you'd use a proper reranking model
            reranked_chunks = []
            
            for chunk in chunks:
                text = chunk["text"]
                # Simple keyword overlap scoring
                query_words = set(query.lower().split())
                text_words = set(text.lower().split())
                overlap = len(query_words.intersection(text_words))
                
                chunk["rerank_score"] = overlap / max(len(query_words), 1)
                reranked_chunks.append(chunk)
            
            # Sort by rerank score
            reranked_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            if self.collection:
                count = self.collection.count()
                return {
                    "total_chunks": count,
                    "vector_db": "ChromaDB",
                    "embedding_model": self.embedding_model,
                    "reranking_model": self.reranking_model
                }
            else:
                return {
                    "total_chunks": len(self.embeddings_cache),
                    "vector_db": "In-memory",
                    "embedding_model": self.embedding_model,
                    "reranking_model": self.reranking_model
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}
