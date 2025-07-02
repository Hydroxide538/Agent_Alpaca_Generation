"""
Enhanced Cleanup Manager for GraphRAG System
Provides comprehensive cleanup functionality for documents, vector DB, graph DB, and results
"""

import os
import logging
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class CleanupManager:
    """Manages comprehensive cleanup operations for the GraphRAG system"""
    
    def __init__(self, 
                 upload_dir: str = "uploads",
                 backend_upload_dir: str = "backend/uploads", 
                 vector_db_dir: str = "vector_db",
                 backend_vector_db_dir: str = "backend/vector_db",
                 results_dir: str = "results",
                 backend_results_dir: str = "backend/results",
                 neo4j_manager=None,
                 graph_rag_system=None):
        
        self.upload_dir = upload_dir
        self.backend_upload_dir = backend_upload_dir
        self.vector_db_dir = vector_db_dir
        self.backend_vector_db_dir = backend_vector_db_dir
        self.results_dir = results_dir
        self.backend_results_dir = backend_results_dir
        self.neo4j_manager = neo4j_manager
        self.graph_rag_system = graph_rag_system
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.upload_dir,
            self.backend_upload_dir,
            self.vector_db_dir,
            self.backend_vector_db_dir,
            self.results_dir,
            self.backend_results_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def clear_queued_documents(self) -> Dict[str, Any]:
        """Clear only uploaded documents waiting for processing"""
        try:
            start_time = datetime.now()
            cleanup_report = {
                'operation': 'clear_queued_documents',
                'started_at': start_time.isoformat(),
                'status': 'success',
                'items_cleared': {},
                'errors': []
            }
            
            # Clear upload directories
            upload_count = await self._clear_directory(self.upload_dir)
            backend_upload_count = await self._clear_directory(self.backend_upload_dir)
            
            cleanup_report['items_cleared'] = {
                'upload_documents': upload_count,
                'backend_upload_documents': backend_upload_count,
                'total_documents': upload_count + backend_upload_count
            }
            
            # Clear document collections (if enhanced document manager is available)
            try:
                if hasattr(self, 'enhanced_document_manager'):
                    collections_cleared = await self.enhanced_document_manager.clear_all_documents()
                    cleanup_report['items_cleared']['collections'] = collections_cleared.get('count', 0)
            except Exception as e:
                cleanup_report['errors'].append(f"Failed to clear document collections: {str(e)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            cleanup_report['completed_at'] = datetime.now().isoformat()
            cleanup_report['processing_time'] = processing_time
            
            logger.info(f"Cleared queued documents in {processing_time:.2f}s")
            return cleanup_report
            
        except Exception as e:
            logger.error(f"Failed to clear queued documents: {str(e)}")
            return {
                'operation': 'clear_queued_documents',
                'status': 'error',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            }
    
    async def fresh_start_cleanup(self) -> Dict[str, Any]:
        """Complete system reset for new dataset generation"""
        try:
            start_time = datetime.now()
            cleanup_report = {
                'operation': 'fresh_start_cleanup',
                'started_at': start_time.isoformat(),
                'status': 'success',
                'items_cleared': {},
                'errors': []
            }
            
            # Step 1: Clear uploaded documents
            upload_count = await self._clear_directory(self.upload_dir)
            backend_upload_count = await self._clear_directory(self.backend_upload_dir)
            
            # Step 2: Clear vector databases
            vector_db_cleared = await self._clear_vector_databases()
            
            # Step 3: Clear Neo4j graph database
            graph_cleared = await self._clear_graph_database()
            
            # Step 4: Clear results
            results_count = await self._clear_directory(self.results_dir)
            backend_results_count = await self._clear_directory(self.backend_results_dir)
            
            # Step 5: Clear logs (optional)
            logs_cleared = await self._clear_logs()
            
            cleanup_report['items_cleared'] = {
                'upload_documents': upload_count,
                'backend_upload_documents': backend_upload_count,
                'vector_databases': vector_db_cleared,
                'graph_database': graph_cleared,
                'results': results_count,
                'backend_results': backend_results_count,
                'logs': logs_cleared,
                'total_documents': upload_count + backend_upload_count,
                'total_results': results_count + backend_results_count
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            cleanup_report['completed_at'] = datetime.now().isoformat()
            cleanup_report['processing_time'] = processing_time
            
            logger.info(f"Fresh start cleanup completed in {processing_time:.2f}s")
            return cleanup_report
            
        except Exception as e:
            logger.error(f"Fresh start cleanup failed: {str(e)}")
            return {
                'operation': 'fresh_start_cleanup',
                'status': 'error',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            }
    
    async def clear_graph_only(self) -> Dict[str, Any]:
        """Clear only the graph database, preserving documents and results"""
        try:
            start_time = datetime.now()
            cleanup_report = {
                'operation': 'clear_graph_only',
                'started_at': start_time.isoformat(),
                'status': 'success',
                'items_cleared': {},
                'errors': []
            }
            
            # Clear Neo4j graph database
            graph_cleared = await self._clear_graph_database()
            
            # Clear vector databases (they're tied to the graph)
            vector_db_cleared = await self._clear_vector_databases()
            
            cleanup_report['items_cleared'] = {
                'graph_database': graph_cleared,
                'vector_databases': vector_db_cleared
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            cleanup_report['completed_at'] = datetime.now().isoformat()
            cleanup_report['processing_time'] = processing_time
            
            logger.info(f"Graph-only cleanup completed in {processing_time:.2f}s")
            return cleanup_report
            
        except Exception as e:
            logger.error(f"Graph-only cleanup failed: {str(e)}")
            return {
                'operation': 'clear_graph_only',
                'status': 'error',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            }
    
    async def optimize_graph(self) -> Dict[str, Any]:
        """Optimize the graph database without full rebuild"""
        try:
            start_time = datetime.now()
            optimization_report = {
                'operation': 'optimize_graph',
                'started_at': start_time.isoformat(),
                'status': 'success',
                'optimizations': {},
                'errors': []
            }
            
            if not self.neo4j_manager or not self.neo4j_manager.is_connected:
                optimization_report['status'] = 'skipped'
                optimization_report['message'] = 'Neo4j not connected'
                return optimization_report
            
            # Get initial statistics
            initial_stats = await self.neo4j_manager.get_graph_statistics()
            
            # Optimization steps
            optimizations = {}
            
            # 1. Remove low-quality entities
            low_quality_removed = await self._remove_low_quality_entities()
            optimizations['low_quality_entities_removed'] = low_quality_removed
            
            # 2. Merge duplicate entities
            duplicates_merged = await self._merge_duplicate_entities()
            optimizations['duplicate_entities_merged'] = duplicates_merged
            
            # 3. Strengthen weak relationships
            relationships_strengthened = await self._strengthen_weak_relationships()
            optimizations['relationships_strengthened'] = relationships_strengthened
            
            # 4. Update quality scores
            quality_updated = await self._update_quality_scores()
            optimizations['quality_scores_updated'] = quality_updated
            
            # Get final statistics
            final_stats = await self.neo4j_manager.get_graph_statistics()
            
            optimization_report['optimizations'] = optimizations
            optimization_report['statistics'] = {
                'before': initial_stats,
                'after': final_stats,
                'improvement': self._calculate_improvement(initial_stats, final_stats)
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            optimization_report['completed_at'] = datetime.now().isoformat()
            optimization_report['processing_time'] = processing_time
            
            logger.info(f"Graph optimization completed in {processing_time:.2f}s")
            return optimization_report
            
        except Exception as e:
            logger.error(f"Graph optimization failed: {str(e)}")
            return {
                'operation': 'optimize_graph',
                'status': 'error',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            }
    
    async def _clear_directory(self, directory_path: str) -> int:
        """Clear all files in a directory and return count"""
        try:
            if not os.path.exists(directory_path):
                return 0
            
            file_count = 0
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    file_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    file_count += 1
            
            return file_count
            
        except Exception as e:
            logger.error(f"Failed to clear directory {directory_path}: {str(e)}")
            return 0
    
    async def _clear_vector_databases(self) -> Dict[str, Any]:
        """Clear vector databases"""
        try:
            cleared = {}
            
            # Clear main vector DB
            if os.path.exists(self.vector_db_dir):
                shutil.rmtree(self.vector_db_dir)
                os.makedirs(self.vector_db_dir, exist_ok=True)
                cleared['main_vector_db'] = True
            
            # Clear backend vector DB
            if os.path.exists(self.backend_vector_db_dir):
                shutil.rmtree(self.backend_vector_db_dir)
                os.makedirs(self.backend_vector_db_dir, exist_ok=True)
                cleared['backend_vector_db'] = True
            
            # Clear GraphRAG vector database if available
            if self.graph_rag_system:
                try:
                    await self.graph_rag_system.clear_all_data()
                    cleared['graphrag_vector_db'] = True
                except Exception as e:
                    logger.warning(f"Failed to clear GraphRAG vector DB: {str(e)}")
                    cleared['graphrag_vector_db'] = False
            
            return cleared
            
        except Exception as e:
            logger.error(f"Failed to clear vector databases: {str(e)}")
            return {'error': str(e)}
    
    async def _clear_graph_database(self) -> Dict[str, Any]:
        """Clear Neo4j graph database"""
        try:
            if not self.neo4j_manager or not self.neo4j_manager.is_connected:
                return {'status': 'skipped', 'reason': 'Neo4j not connected'}
            
            # Get statistics before clearing
            stats_before = await self.neo4j_manager.get_graph_statistics()
            
            # Clear the graph
            success = await self.neo4j_manager.clear_graph()
            
            if success:
                return {
                    'status': 'success',
                    'entities_cleared': stats_before.get('total_entities', 0),
                    'relationships_cleared': stats_before.get('total_relationships', 0),
                    'documents_cleared': stats_before.get('total_documents', 0)
                }
            else:
                return {'status': 'failed', 'reason': 'Clear operation failed'}
                
        except Exception as e:
            logger.error(f"Failed to clear graph database: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def _clear_logs(self) -> Dict[str, Any]:
        """Clear log files (optional)"""
        try:
            cleared = {}
            
            # Clear logs directory if it exists
            logs_dir = "logs"
            if os.path.exists(logs_dir):
                log_count = await self._clear_directory(logs_dir)
                cleared['log_files'] = log_count
            
            return cleared
            
        except Exception as e:
            logger.error(f"Failed to clear logs: {str(e)}")
            return {'error': str(e)}
    
    async def _remove_low_quality_entities(self) -> int:
        """Remove entities with low quality scores"""
        try:
            if not self.neo4j_manager or not self.neo4j_manager.is_connected:
                return 0
            
            # This would require custom Cypher queries
            # For now, return 0 as placeholder
            return 0
            
        except Exception as e:
            logger.error(f"Failed to remove low quality entities: {str(e)}")
            return 0
    
    async def _merge_duplicate_entities(self) -> int:
        """Merge duplicate entities"""
        try:
            if not self.neo4j_manager or not self.neo4j_manager.is_connected:
                return 0
            
            # This would require sophisticated duplicate detection
            # For now, return 0 as placeholder
            return 0
            
        except Exception as e:
            logger.error(f"Failed to merge duplicate entities: {str(e)}")
            return 0
    
    async def _strengthen_weak_relationships(self) -> int:
        """Strengthen weak relationships with additional evidence"""
        try:
            if not self.neo4j_manager or not self.neo4j_manager.is_connected:
                return 0
            
            # This would require relationship analysis
            # For now, return 0 as placeholder
            return 0
            
        except Exception as e:
            logger.error(f"Failed to strengthen weak relationships: {str(e)}")
            return 0
    
    async def _update_quality_scores(self) -> int:
        """Update quality scores for all entities and relationships"""
        try:
            if not self.neo4j_manager or not self.neo4j_manager.is_connected:
                return 0
            
            # This would require quality score recalculation
            # For now, return 0 as placeholder
            return 0
            
        except Exception as e:
            logger.error(f"Failed to update quality scores: {str(e)}")
            return 0
    
    def _calculate_improvement(self, before_stats: Dict[str, Any], after_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement metrics"""
        try:
            improvement = {}
            
            # Entity count change
            entities_before = before_stats.get('total_entities', 0)
            entities_after = after_stats.get('total_entities', 0)
            improvement['entity_count_change'] = entities_after - entities_before
            
            # Relationship count change
            rels_before = before_stats.get('total_relationships', 0)
            rels_after = after_stats.get('total_relationships', 0)
            improvement['relationship_count_change'] = rels_after - rels_before
            
            # Quality improvement
            quality_before = before_stats.get('quality_metrics', {}).get('average_quality_score', 0.0)
            quality_after = after_stats.get('quality_metrics', {}).get('average_quality_score', 0.0)
            improvement['quality_score_improvement'] = quality_after - quality_before
            
            # Density improvement
            density_before = before_stats.get('graph_density', 0.0)
            density_after = after_stats.get('graph_density', 0.0)
            improvement['density_improvement'] = density_after - density_before
            
            return improvement
            
        except Exception as e:
            logger.error(f"Failed to calculate improvement: {str(e)}")
            return {}
    
    async def get_cleanup_status(self) -> Dict[str, Any]:
        """Get current status of all cleanable items"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'directories': {},
                'databases': {},
                'total_items': 0
            }
            
            # Check directories
            directories = {
                'uploads': self.upload_dir,
                'backend_uploads': self.backend_upload_dir,
                'vector_db': self.vector_db_dir,
                'backend_vector_db': self.backend_vector_db_dir,
                'results': self.results_dir,
                'backend_results': self.backend_results_dir
            }
            
            total_files = 0
            for name, path in directories.items():
                if os.path.exists(path):
                    file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                    dir_count = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
                    status['directories'][name] = {
                        'path': path,
                        'files': file_count,
                        'directories': dir_count,
                        'total_items': file_count + dir_count
                    }
                    total_files += file_count + dir_count
                else:
                    status['directories'][name] = {
                        'path': path,
                        'files': 0,
                        'directories': 0,
                        'total_items': 0,
                        'exists': False
                    }
            
            # Check databases
            if self.neo4j_manager and self.neo4j_manager.is_connected:
                try:
                    graph_stats = await self.neo4j_manager.get_graph_statistics()
                    status['databases']['neo4j'] = {
                        'connected': True,
                        'entities': graph_stats.get('total_entities', 0),
                        'relationships': graph_stats.get('total_relationships', 0),
                        'documents': graph_stats.get('total_documents', 0)
                    }
                except Exception as e:
                    status['databases']['neo4j'] = {
                        'connected': False,
                        'error': str(e)
                    }
            else:
                status['databases']['neo4j'] = {
                    'connected': False,
                    'reason': 'Not initialized or not connected'
                }
            
            # Check GraphRAG system
            if self.graph_rag_system:
                try:
                    graphrag_stats = await self.graph_rag_system.get_graph_statistics()
                    status['databases']['graphrag'] = {
                        'available': True,
                        'vector_documents': graphrag_stats.get('vector_documents', 0)
                    }
                except Exception as e:
                    status['databases']['graphrag'] = {
                        'available': False,
                        'error': str(e)
                    }
            else:
                status['databases']['graphrag'] = {
                    'available': False,
                    'reason': 'Not initialized'
                }
            
            status['total_items'] = total_files
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get cleanup status: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def estimate_cleanup_impact(self, operation: str) -> Dict[str, Any]:
        """Estimate the impact of a cleanup operation"""
        try:
            current_status = await self.get_cleanup_status()
            
            impact = {
                'operation': operation,
                'estimated_items_affected': 0,
                'estimated_data_size': '0 MB',
                'estimated_time': '< 1 minute',
                'reversible': False,
                'warnings': []
            }
            
            if operation == 'clear_queued_documents':
                upload_items = current_status['directories'].get('uploads', {}).get('total_items', 0)
                backend_upload_items = current_status['directories'].get('backend_uploads', {}).get('total_items', 0)
                impact['estimated_items_affected'] = upload_items + backend_upload_items
                impact['reversible'] = False
                impact['warnings'] = ['Uploaded documents will be permanently deleted']
                
            elif operation == 'fresh_start_cleanup':
                total_items = current_status['total_items']
                neo4j_items = 0
                if current_status['databases'].get('neo4j', {}).get('connected'):
                    neo4j_stats = current_status['databases']['neo4j']
                    neo4j_items = (neo4j_stats.get('entities', 0) + 
                                 neo4j_stats.get('relationships', 0) + 
                                 neo4j_stats.get('documents', 0))
                
                impact['estimated_items_affected'] = total_items + neo4j_items
                impact['estimated_time'] = '1-3 minutes'
                impact['reversible'] = False
                impact['warnings'] = [
                    'ALL data will be permanently deleted',
                    'This includes documents, results, and knowledge graph',
                    'This operation cannot be undone'
                ]
                
            elif operation == 'clear_graph_only':
                neo4j_items = 0
                if current_status['databases'].get('neo4j', {}).get('connected'):
                    neo4j_stats = current_status['databases']['neo4j']
                    neo4j_items = (neo4j_stats.get('entities', 0) + 
                                 neo4j_stats.get('relationships', 0))
                
                impact['estimated_items_affected'] = neo4j_items
                impact['reversible'] = True
                impact['warnings'] = ['Knowledge graph will be cleared but can be rebuilt from documents']
                
            elif operation == 'optimize_graph':
                neo4j_items = 0
                if current_status['databases'].get('neo4j', {}).get('connected'):
                    neo4j_stats = current_status['databases']['neo4j']
                    neo4j_items = neo4j_stats.get('entities', 0)
                
                impact['estimated_items_affected'] = neo4j_items
                impact['estimated_time'] = '2-5 minutes'
                impact['reversible'] = True
                impact['warnings'] = ['Graph structure may change but data will be preserved']
            
            return impact
            
        except Exception as e:
            logger.error(f"Failed to estimate cleanup impact: {str(e)}")
            return {
                'operation': operation,
                'error': str(e)
            }
