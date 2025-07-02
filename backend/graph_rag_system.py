"""
GraphRAG System - Enhanced RAG with Knowledge Graph Integration
Combines traditional vector search with graph-based contextual retrieval
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio
import uuid
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from backend.neo4j_manager import Neo4jManager
from backend.entity_extraction_crew import EntityExtractionCrew, EntityExtractionResult
from backend.rag_system import RAGSystem

logger = logging.getLogger(__name__)

@dataclass
class GraphRAGResult:
    """Result from GraphRAG query"""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    graph_context: Dict[str, Any]
    vector_context: List[Dict[str, Any]]
    reasoning_path: List[str]
    quality_score: float
    processing_time: float

class GraphRAGSystem:
    """Enhanced RAG system with knowledge graph integration"""
    
    def __init__(self, 
                 llm_manager,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password",
                 vector_db_path: str = "./vector_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.llm_manager = llm_manager
        self.neo4j_manager = Neo4jManager(neo4j_uri, neo4j_user, neo4j_password)
        self.entity_extraction_crew = EntityExtractionCrew(llm_manager, self.neo4j_manager)
        
        # Initialize vector database
        self.vector_db_path = vector_db_path
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = None
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        # Traditional RAG system for fallback
        self.traditional_rag = RAGSystem()
        
        # Quality thresholds
        self.min_graph_quality = 0.6
        self.min_entity_confidence = 0.5
        self.min_relationship_strength = 0.4
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            
            # Initialize vector collection
            try:
                self.collection = self.chroma_client.get_collection("graphrag_documents")
                logger.info("Connected to existing vector collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="graphrag_documents",
                    metadata={"description": "GraphRAG document embeddings"}
                )
                logger.info("Created new vector collection")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG components: {str(e)}")
            raise
    
    async def connect(self) -> bool:
        """Connect to all backend services"""
        try:
            # Connect to Neo4j
            neo4j_connected = await self.neo4j_manager.connect()
            if not neo4j_connected:
                logger.warning("Neo4j connection failed - GraphRAG will use traditional RAG only")
                return False
            
            logger.info("GraphRAG system connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect GraphRAG system: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from all backend services"""
        try:
            await self.neo4j_manager.disconnect()
            logger.info("GraphRAG system disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting GraphRAG system: {str(e)}")
    
    async def process_document(self, 
                             document_text: str, 
                             document_id: str, 
                             document_metadata: Dict[str, Any] = None) -> EntityExtractionResult:
        """Process a document through the GraphRAG pipeline"""
        try:
            start_time = datetime.now()
            
            # Get LLMs for processing
            manager_llm = await self.llm_manager.get_manager_llm()
            qwen_llm = await self.llm_manager.get_qwen_llm()  # Specialized for entity extraction
            
            if not manager_llm:
                raise Exception("Manager LLM not available")
            
            # Use manager LLM as fallback if Qwen not available
            if not qwen_llm:
                logger.warning("Qwen LLM not available, using manager LLM for entity extraction")
                qwen_llm = manager_llm
            
            # Step 1: Extract entities and relationships using CrewAI
            logger.info(f"Starting entity extraction for document {document_id}")
            extraction_result = await self.entity_extraction_crew.process_document(
                document_text, document_id, manager_llm, qwen_llm
            )
            
            # Step 2: Store document in vector database
            await self._store_document_vector(document_text, document_id, document_metadata)
            
            # Step 3: Create document chunks for better retrieval
            chunks = self._create_document_chunks(document_text, document_id)
            await self._store_chunks_vector(chunks)
            
            # Step 4: Update graph quality metrics
            await self._update_graph_quality_metrics()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            extraction_result.processing_time = processing_time
            
            logger.info(f"Document processing completed in {processing_time:.2f}s")
            logger.info(f"Extracted {len(extraction_result.entities)} entities and {len(extraction_result.relationships)} relationships")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return EntityExtractionResult(
                entities=[],
                relationships=[],
                quality_score=0.0,
                confidence=0.0,
                processing_time=0.0,
                document_id=document_id,
                metadata={'error': str(e)}
            )
    
    async def query(self, 
                   query_text: str, 
                   max_results: int = 5,
                   use_graph_expansion: bool = True,
                   graph_depth: int = 2) -> GraphRAGResult:
        """Query the GraphRAG system"""
        try:
            start_time = datetime.now()
            
            # Step 1: Extract entities from query
            query_entities = await self._extract_query_entities(query_text)
            
            # Step 2: Get graph context if entities found and graph available
            graph_context = {}
            if query_entities and self.neo4j_manager.is_connected and use_graph_expansion:
                graph_context = await self._get_graph_context(query_entities, graph_depth)
            
            # Step 3: Get vector similarity results
            vector_context = await self._get_vector_context(query_text, max_results)
            
            # Step 4: Combine contexts and generate answer
            answer, confidence, reasoning_path = await self._generate_answer(
                query_text, graph_context, vector_context
            )
            
            # Step 5: Calculate quality score
            quality_score = self._calculate_result_quality(graph_context, vector_context, confidence)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return GraphRAGResult(
                answer=answer,
                confidence=confidence,
                sources=self._extract_sources(graph_context, vector_context),
                graph_context=graph_context,
                vector_context=vector_context,
                reasoning_path=reasoning_path,
                quality_score=quality_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"GraphRAG query failed: {str(e)}")
            return GraphRAGResult(
                answer=f"Query failed: {str(e)}",
                confidence=0.0,
                sources=[],
                graph_context={},
                vector_context=[],
                reasoning_path=[f"Error: {str(e)}"],
                quality_score=0.0,
                processing_time=0.0
            )
    
    async def _extract_query_entities(self, query_text: str) -> List[Dict[str, Any]]:
        """Extract entities from the query text"""
        try:
            # Use the entity extraction tool
            tool = self.entity_extraction_crew.tools['entity_extraction']
            result = tool._run(query_text, "query")
            
            if result['success']:
                return result['entities']
            else:
                logger.warning(f"Query entity extraction failed: {result['message']}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract query entities: {str(e)}")
            return []
    
    async def _get_graph_context(self, query_entities: List[Dict[str, Any]], depth: int = 2) -> Dict[str, Any]:
        """Get graph context for query entities"""
        try:
            graph_context = {
                'entities': {},
                'relationships': [],
                'paths': [],
                'statistics': {}
            }
            
            # Find matching entities in the graph
            for query_entity in query_entities:
                entity_name = query_entity.get('name', '')
                entity_type = query_entity.get('type', '')
                
                # Search for similar entities in the graph
                matching_entities = await self.neo4j_manager.search_entities(
                    entity_name, entity_type, limit=10
                )
                
                for entity in matching_entities:
                    entity_id = entity['id']
                    
                    # Get entity relationships
                    entity_graph = await self.neo4j_manager.get_entity_relationships(
                        entity_id, depth
                    )
                    
                    # Merge into graph context
                    graph_context['entities'].update(entity_graph['entities'])
                    graph_context['relationships'].extend(entity_graph['relationships'])
            
            # Remove duplicates from relationships
            unique_relationships = {}
            for rel in graph_context['relationships']:
                key = (rel.get('source'), rel.get('target'), rel.get('type'))
                if key not in unique_relationships:
                    unique_relationships[key] = rel
            
            graph_context['relationships'] = list(unique_relationships.values())
            
            # Calculate statistics
            graph_context['statistics'] = {
                'entity_count': len(graph_context['entities']),
                'relationship_count': len(graph_context['relationships']),
                'avg_confidence': self._calculate_avg_confidence(graph_context['entities']),
                'coverage_score': self._calculate_coverage_score(query_entities, graph_context['entities'])
            }
            
            return graph_context
            
        except Exception as e:
            logger.error(f"Failed to get graph context: {str(e)}")
            return {'entities': {}, 'relationships': [], 'paths': [], 'statistics': {}}
    
    async def _get_vector_context(self, query_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get vector similarity context"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query_text])[0].tolist()
            
            # Query vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            vector_context = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    context_item = {
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'rank': i + 1
                    }
                    vector_context.append(context_item)
            
            return vector_context
            
        except Exception as e:
            logger.error(f"Failed to get vector context: {str(e)}")
            return []
    
    async def _generate_answer(self, 
                             query_text: str, 
                             graph_context: Dict[str, Any], 
                             vector_context: List[Dict[str, Any]]) -> Tuple[str, float, List[str]]:
        """Generate answer using combined graph and vector context"""
        try:
            # Get manager LLM
            manager_llm = await self.llm_manager.get_manager_llm()
            if not manager_llm:
                raise Exception("Manager LLM not available")
            
            # Prepare context for LLM
            context_prompt = self._build_context_prompt(query_text, graph_context, vector_context)
            
            # Generate answer
            response = await manager_llm.agenerate([context_prompt])
            answer = response.generations[0][0].text.strip()
            
            # Calculate confidence based on context quality
            confidence = self._calculate_answer_confidence(graph_context, vector_context, answer)
            
            # Generate reasoning path
            reasoning_path = self._generate_reasoning_path(graph_context, vector_context)
            
            return answer, confidence, reasoning_path
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return f"Failed to generate answer: {str(e)}", 0.0, [f"Error: {str(e)}"]
    
    def _build_context_prompt(self, 
                            query_text: str, 
                            graph_context: Dict[str, Any], 
                            vector_context: List[Dict[str, Any]]) -> str:
        """Build comprehensive context prompt for LLM"""
        
        prompt_parts = [
            "You are an advanced AI assistant with access to both knowledge graph and document information.",
            "Use the provided context to answer the user's question accurately and comprehensively.",
            "",
            f"USER QUESTION: {query_text}",
            ""
        ]
        
        # Add graph context
        if graph_context.get('entities'):
            prompt_parts.extend([
                "KNOWLEDGE GRAPH CONTEXT:",
                "Relevant entities and their relationships:",
                ""
            ])
            
            # Add entity information
            for entity_id, entity in graph_context['entities'].items():
                entity_info = f"- {entity.get('name', 'Unknown')} ({entity.get('type', 'Entity')})"
                if entity.get('attributes'):
                    entity_info += f" - {entity['attributes']}"
                prompt_parts.append(entity_info)
            
            prompt_parts.append("")
            
            # Add relationship information
            if graph_context.get('relationships'):
                prompt_parts.append("Relationships:")
                for rel in graph_context['relationships'][:10]:  # Limit to top 10
                    source_entity = graph_context['entities'].get(rel.get('source'), {})
                    target_entity = graph_context['entities'].get(rel.get('target'), {})
                    rel_info = f"- {source_entity.get('name', 'Unknown')} {rel.get('type', 'RELATED_TO')} {target_entity.get('name', 'Unknown')}"
                    if rel.get('context'):
                        rel_info += f" (Context: {rel['context'][:100]}...)"
                    prompt_parts.append(rel_info)
                
                prompt_parts.append("")
        
        # Add vector context
        if vector_context:
            prompt_parts.extend([
                "DOCUMENT CONTEXT:",
                "Relevant document excerpts:",
                ""
            ])
            
            for i, context in enumerate(vector_context[:3]):  # Top 3 most relevant
                prompt_parts.append(f"Document {i+1} (Similarity: {context['similarity']:.2f}):")
                prompt_parts.append(context['document'][:500] + "..." if len(context['document']) > 500 else context['document'])
                prompt_parts.append("")
        
        # Add instructions
        prompt_parts.extend([
            "INSTRUCTIONS:",
            "1. Use both the knowledge graph and document context to provide a comprehensive answer",
            "2. Cite specific entities, relationships, or documents when relevant",
            "3. If the context doesn't contain enough information, say so clearly",
            "4. Provide a confidence level for your answer (High/Medium/Low)",
            "5. Be concise but thorough",
            "",
            "ANSWER:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _calculate_answer_confidence(self, 
                                   graph_context: Dict[str, Any], 
                                   vector_context: List[Dict[str, Any]], 
                                   answer: str) -> float:
        """Calculate confidence score for the generated answer"""
        
        confidence_factors = []
        
        # Graph context quality
        if graph_context.get('entities'):
            graph_quality = graph_context.get('statistics', {}).get('avg_confidence', 0.0)
            entity_count = len(graph_context['entities'])
            relationship_count = len(graph_context.get('relationships', []))
            
            graph_score = (graph_quality * 0.5 + 
                          min(entity_count / 10, 1.0) * 0.3 + 
                          min(relationship_count / 5, 1.0) * 0.2)
            confidence_factors.append(graph_score)
        
        # Vector context quality
        if vector_context:
            avg_similarity = sum(ctx['similarity'] for ctx in vector_context) / len(vector_context)
            vector_score = avg_similarity
            confidence_factors.append(vector_score)
        
        # Answer quality heuristics
        answer_length = len(answer.split())
        length_score = min(answer_length / 50, 1.0)  # Optimal around 50 words
        confidence_factors.append(length_score)
        
        # Overall confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default medium confidence
    
    def _generate_reasoning_path(self, 
                               graph_context: Dict[str, Any], 
                               vector_context: List[Dict[str, Any]]) -> List[str]:
        """Generate reasoning path for transparency"""
        
        path = []
        
        if graph_context.get('entities'):
            entity_count = len(graph_context['entities'])
            rel_count = len(graph_context.get('relationships', []))
            path.append(f"Found {entity_count} relevant entities and {rel_count} relationships in knowledge graph")
        
        if vector_context:
            path.append(f"Retrieved {len(vector_context)} relevant document excerpts")
            avg_sim = sum(ctx['similarity'] for ctx in vector_context) / len(vector_context)
            path.append(f"Average document similarity: {avg_sim:.2f}")
        
        if graph_context.get('entities') and vector_context:
            path.append("Combined graph and vector contexts for comprehensive answer")
        elif graph_context.get('entities'):
            path.append("Used primarily knowledge graph context")
        elif vector_context:
            path.append("Used primarily document vector context")
        else:
            path.append("Limited context available - answer may be incomplete")
        
        return path
    
    def _calculate_result_quality(self, 
                                graph_context: Dict[str, Any], 
                                vector_context: List[Dict[str, Any]], 
                                confidence: float) -> float:
        """Calculate overall result quality score"""
        
        quality_factors = []
        
        # Context availability
        has_graph = bool(graph_context.get('entities'))
        has_vector = bool(vector_context)
        
        if has_graph and has_vector:
            context_score = 1.0
        elif has_graph or has_vector:
            context_score = 0.7
        else:
            context_score = 0.3
        
        quality_factors.append(context_score)
        
        # Graph quality
        if has_graph:
            graph_stats = graph_context.get('statistics', {})
            graph_quality = graph_stats.get('avg_confidence', 0.5)
            quality_factors.append(graph_quality)
        
        # Vector quality
        if has_vector:
            avg_similarity = sum(ctx['similarity'] for ctx in vector_context) / len(vector_context)
            quality_factors.append(avg_similarity)
        
        # Answer confidence
        quality_factors.append(confidence)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _extract_sources(self, 
                        graph_context: Dict[str, Any], 
                        vector_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information for citation"""
        
        sources = []
        
        # Graph sources
        if graph_context.get('entities'):
            for entity_id, entity in graph_context['entities'].items():
                source_doc = entity.get('source_document')
                if source_doc:
                    sources.append({
                        'type': 'graph_entity',
                        'entity_name': entity.get('name'),
                        'entity_type': entity.get('type'),
                        'source_document': source_doc,
                        'confidence': entity.get('confidence', 0.0)
                    })
        
        # Vector sources
        for ctx in vector_context:
            metadata = ctx.get('metadata', {})
            sources.append({
                'type': 'document_excerpt',
                'document_id': metadata.get('document_id'),
                'similarity': ctx['similarity'],
                'excerpt': ctx['document'][:200] + "..." if len(ctx['document']) > 200 else ctx['document']
            })
        
        return sources
    
    async def _store_document_vector(self, 
                                   document_text: str, 
                                   document_id: str, 
                                   metadata: Dict[str, Any] = None):
        """Store document in vector database"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([document_text])[0].tolist()
            
            # Prepare metadata
            doc_metadata = {
                'document_id': document_id,
                'created_at': datetime.now().isoformat(),
                'text_length': len(document_text),
                'type': 'full_document'
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Store in vector database
            self.collection.add(
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[doc_metadata],
                ids=[document_id]
            )
            
            logger.debug(f"Stored document {document_id} in vector database")
            
        except Exception as e:
            logger.error(f"Failed to store document vector: {str(e)}")
            raise
    
    def _create_document_chunks(self, document_text: str, document_id: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """Create document chunks for better retrieval"""
        
        chunks = []
        words = document_text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk = {
                'id': f"{document_id}_chunk_{i // chunk_size}",
                'text': chunk_text,
                'document_id': document_id,
                'chunk_index': i // chunk_size,
                'start_word': i,
                'end_word': min(i + chunk_size, len(words))
            }
            chunks.append(chunk)
        
        return chunks
    
    async def _store_chunks_vector(self, chunks: List[Dict[str, Any]]):
        """Store document chunks in vector database"""
        try:
            if not chunks:
                return
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(chunk_texts).tolist()
            
            # Prepare data for storage
            ids = [chunk['id'] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    'document_id': chunk['document_id'],
                    'chunk_index': chunk['chunk_index'],
                    'start_word': chunk['start_word'],
                    'end_word': chunk['end_word'],
                    'type': 'document_chunk',
                    'created_at': datetime.now().isoformat()
                }
                metadatas.append(metadata)
            
            # Store in vector database
            self.collection.add(
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.debug(f"Stored {len(chunks)} document chunks in vector database")
            
        except Exception as e:
            logger.error(f"Failed to store document chunks: {str(e)}")
            raise
    
    async def _update_graph_quality_metrics(self):
        """Update overall graph quality metrics"""
        try:
            if not self.neo4j_manager.is_connected:
                return
            
            stats = await self.neo4j_manager.get_graph_statistics()
            
            # Store quality metrics for monitoring
            quality_metrics = {
                'timestamp': datetime.now().isoformat(),
                'total_entities': stats.get('total_entities', 0),
                'total_relationships': stats.get('total_relationships', 0),
                'graph_density': stats.get('graph_density', 0.0),
                'average_quality': stats.get('quality_metrics', {}).get('average_quality_score', 0.0)
            }
            
            logger.debug(f"Updated graph quality metrics: {quality_metrics}")
            
        except Exception as e:
            logger.error(f"Failed to update graph quality metrics: {str(e)}")
    
    def _calculate_avg_confidence(self, entities: Dict[str, Any]) -> float:
        """Calculate average confidence of entities"""
        if not entities:
            return 0.0
        
        confidences = [entity.get('confidence', 0.0) for entity in entities.values()]
        return sum(confidences) / len(confidences)
    
    def _calculate_coverage_score(self, query_entities: List[Dict[str, Any]], graph_entities: Dict[str, Any]) -> float:
        """Calculate how well graph entities cover query entities"""
        if not query_entities:
            return 1.0
        
        if not graph_entities:
            return 0.0
        
        query_names = set(entity.get('name', '').lower() for entity in query_entities)
        graph_names = set(entity.get('name', '').lower() for entity in graph_entities.values())
        
        if not query_names:
            return 1.0
        
        covered = len(query_names.intersection(graph_names))
        return covered / len(query_names)
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            if not self.neo4j_manager.is_connected:
                return {'error': 'Neo4j not connected'}
            
            stats = await self.neo4j_manager.get_graph_statistics()
            
            # Add vector database statistics
            try:
                vector_count = self.collection.count()
                stats['vector_documents'] = vector_count
            except:
                stats['vector_documents'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {str(e)}")
            return {'error': str(e)}
    
    async def get_visualization_data(self, limit: int = 500) -> Dict[str, Any]:
        """Get graph data for visualization"""
        try:
            if not self.neo4j_manager.is_connected:
                return {'error': 'Neo4j not connected'}
            
            return await self.neo4j_manager.get_visualization_data(limit)
            
        except Exception as e:
            logger.error(f"Failed to get visualization data: {str(e)}")
            return {'error': str(e)}
    
    async def clear_all_data(self) -> bool:
        """Clear all graph and vector data"""
        try:
            # Clear Neo4j graph
            if self.neo4j_manager.is_connected:
                await self.neo4j_manager.clear_graph()
            
            # Clear vector database
            try:
                self.chroma_client.delete_collection("graphrag_documents")
                self.collection = self.chroma_client.create_collection(
                    name="graphrag_documents",
                    metadata={"description": "GraphRAG document embeddings"}
                )
            except:
                pass
            
            logger.info("Cleared all GraphRAG data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear GraphRAG data: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'components': {}
            }
            
            # Check Neo4j
            neo4j_health = await self.neo4j_manager.health_check()
            health_status['components']['neo4j'] = neo4j_health
            
            # Check vector database
            try:
                vector_count = self.collection.count()
                health_status['components']['vector_db'] = {
                    'status': 'healthy',
                    'document_count': vector_count
                }
            except Exception as e:
                health_status['components']['vector_db'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Check embedding model
            try:
                test_embedding = self.embedding_model.encode(["test"])
                health_status['components']['embedding_model'] = {
                    'status': 'healthy',
                    'model_name': self.embedding_model_name,
                    'embedding_dimension': len(test_embedding[0])
                }
            except Exception as e:
                health_status['components']['embedding_model'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Determine overall status
            component_statuses = [comp.get('status') for comp in health_status['components'].values()]
            if 'error' in component_statuses:
                health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }
