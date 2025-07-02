"""
Neo4j Graph Database Manager for GraphRAG Implementation
Handles all Neo4j operations including entity and relationship management
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import uuid

logger = logging.getLogger(__name__)

class Neo4jManager:
    """Manages Neo4j database operations for GraphRAG"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            # Test connection
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.consume()
            
            self.is_connected = True
            logger.info("Successfully connected to Neo4j database")
            
            # Initialize schema
            await self._initialize_schema()
            
            return True
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.is_connected = False
            logger.info("Disconnected from Neo4j database")
    
    async def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        try:
            async with self.driver.session() as session:
                # Create constraints for unique entities
                constraints = [
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT place_id_unique IF NOT EXISTS FOR (p:Place) REQUIRE p.id IS UNIQUE",
                    "CREATE CONSTRAINT thing_id_unique IF NOT EXISTS FOR (t:Thing) REQUIRE t.id IS UNIQUE",
                    "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        await session.run(constraint)
                    except Exception as e:
                        # Constraint might already exist
                        logger.debug(f"Constraint creation note: {str(e)}")
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX document_name_index IF NOT EXISTS FOR (d:Document) ON (d.name)",
                    "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r]-() ON (r.type)",
                    "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)"
                ]
                
                for index in indexes:
                    try:
                        await session.run(index)
                    except Exception as e:
                        logger.debug(f"Index creation note: {str(e)}")
                
                logger.info("Neo4j schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {str(e)}")
            raise
    
    async def create_entity(self, entity_data: Dict[str, Any]) -> str:
        """Create an entity node in the graph"""
        try:
            entity_id = entity_data.get('id', str(uuid.uuid4()))
            entity_type = entity_data.get('type', 'Entity')
            
            # Prepare entity properties
            properties = {
                'id': entity_id,
                'name': entity_data.get('name', ''),
                'type': entity_data.get('type', 'Entity'),
                'confidence': entity_data.get('confidence', 0.0),
                'source_document': entity_data.get('source_document', ''),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'attributes': json.dumps(entity_data.get('attributes', {})),
                'mentions': entity_data.get('mentions', 1),
                'quality_score': entity_data.get('quality_score', 0.0)
            }
            
            # Create entity with appropriate label
            query = f"""
            MERGE (e:{entity_type}:Entity {{id: $id}})
            SET e += $properties
            RETURN e.id as entity_id
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, id=entity_id, properties=properties)
                record = await result.single()
                
                if record:
                    logger.debug(f"Created entity: {entity_id} ({entity_type})")
                    return record['entity_id']
                else:
                    raise Exception("Failed to create entity")
                    
        except Exception as e:
            logger.error(f"Failed to create entity: {str(e)}")
            raise
    
    async def create_relationship(self, source_id: str, target_id: str, relationship_data: Dict[str, Any]) -> bool:
        """Create a relationship between two entities"""
        try:
            relationship_type = relationship_data.get('type', 'RELATED_TO')
            
            properties = {
                'type': relationship_type,
                'confidence': relationship_data.get('confidence', 0.0),
                'strength': relationship_data.get('strength', 0.0),
                'source_document': relationship_data.get('source_document', ''),
                'context': relationship_data.get('context', ''),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'quality_score': relationship_data.get('quality_score', 0.0),
                'validated': relationship_data.get('validated', False)
            }
            
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            MERGE (source)-[r:{relationship_type}]->(target)
            SET r += $properties
            RETURN r
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, 
                                         source_id=source_id, 
                                         target_id=target_id, 
                                         properties=properties)
                record = await result.single()
                
                if record:
                    logger.debug(f"Created relationship: {source_id} -{relationship_type}-> {target_id}")
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to create relationship: {str(e)}")
            return False
    
    async def create_document_node(self, document_data: Dict[str, Any]) -> str:
        """Create a document node in the graph"""
        try:
            doc_id = document_data.get('id', str(uuid.uuid4()))
            
            properties = {
                'id': doc_id,
                'name': document_data.get('name', ''),
                'path': document_data.get('path', ''),
                'type': document_data.get('type', 'document'),
                'size': document_data.get('size', 0),
                'token_count': document_data.get('token_count', 0),
                'created_at': datetime.now().isoformat(),
                'processed_at': datetime.now().isoformat(),
                'metadata': json.dumps(document_data.get('metadata', {}))
            }
            
            query = """
            MERGE (d:Document {id: $id})
            SET d += $properties
            RETURN d.id as document_id
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, id=doc_id, properties=properties)
                record = await result.single()
                
                if record:
                    logger.debug(f"Created document node: {doc_id}")
                    return record['document_id']
                else:
                    raise Exception("Failed to create document node")
                    
        except Exception as e:
            logger.error(f"Failed to create document node: {str(e)}")
            raise
    
    async def link_entity_to_document(self, entity_id: str, document_id: str, context: str = "") -> bool:
        """Link an entity to its source document"""
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            MATCH (d:Document {id: $document_id})
            MERGE (e)-[r:MENTIONED_IN]->(d)
            SET r.context = $context, r.created_at = $timestamp
            RETURN r
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, 
                                         entity_id=entity_id, 
                                         document_id=document_id,
                                         context=context,
                                         timestamp=datetime.now().isoformat())
                record = await result.single()
                return record is not None
                
        except Exception as e:
            logger.error(f"Failed to link entity to document: {str(e)}")
            return False
    
    async def get_entities_by_type(self, entity_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve entities by type"""
        try:
            if entity_type:
                query = f"""
                MATCH (e:{entity_type})
                RETURN e
                ORDER BY e.quality_score DESC, e.mentions DESC
                LIMIT $limit
                """
            else:
                query = """
                MATCH (e:Entity)
                RETURN e
                ORDER BY e.quality_score DESC, e.mentions DESC
                LIMIT $limit
                """
            
            async with self.driver.session() as session:
                result = await session.run(query, limit=limit)
                entities = []
                
                async for record in result:
                    entity = dict(record['e'])
                    # Parse JSON attributes
                    if 'attributes' in entity:
                        try:
                            entity['attributes'] = json.loads(entity['attributes'])
                        except:
                            entity['attributes'] = {}
                    entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to get entities by type: {str(e)}")
            return []
    
    async def get_entity_relationships(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get all relationships for an entity up to max_depth"""
        try:
            query = """
            MATCH path = (start:Entity {id: $entity_id})-[*1..$max_depth]-(connected:Entity)
            RETURN path
            LIMIT 1000
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, entity_id=entity_id, max_depth=max_depth)
                
                relationships = []
                entities = {}
                
                async for record in result:
                    path = record['path']
                    
                    # Extract nodes and relationships from path
                    for i, node in enumerate(path.nodes):
                        node_data = dict(node)
                        if 'attributes' in node_data:
                            try:
                                node_data['attributes'] = json.loads(node_data['attributes'])
                            except:
                                node_data['attributes'] = {}
                        entities[node_data['id']] = node_data
                    
                    for relationship in path.relationships:
                        rel_data = dict(relationship)
                        rel_data['source'] = rel_data.get('start_node_id')
                        rel_data['target'] = rel_data.get('end_node_id')
                        relationships.append(rel_data)
                
                return {
                    'entities': entities,
                    'relationships': relationships,
                    'center_entity': entity_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get entity relationships: {str(e)}")
            return {'entities': {}, 'relationships': [], 'center_entity': entity_id}
    
    async def search_entities(self, search_term: str, entity_type: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Search entities by name or attributes"""
        try:
            if entity_type:
                query = f"""
                MATCH (e:{entity_type})
                WHERE toLower(e.name) CONTAINS toLower($search_term)
                   OR toLower(e.attributes) CONTAINS toLower($search_term)
                RETURN e
                ORDER BY e.quality_score DESC, e.confidence DESC
                LIMIT $limit
                """
            else:
                query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($search_term)
                   OR toLower(e.attributes) CONTAINS toLower($search_term)
                RETURN e
                ORDER BY e.quality_score DESC, e.confidence DESC
                LIMIT $limit
                """
            
            async with self.driver.session() as session:
                result = await session.run(query, search_term=search_term, limit=limit)
                entities = []
                
                async for record in result:
                    entity = dict(record['e'])
                    if 'attributes' in entity:
                        try:
                            entity['attributes'] = json.loads(entity['attributes'])
                        except:
                            entity['attributes'] = {}
                    entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to search entities: {str(e)}")
            return []
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            async with self.driver.session() as session:
                # Entity counts by type
                entity_stats = await session.run("""
                MATCH (e:Entity)
                RETURN e.type as entity_type, count(e) as count
                ORDER BY count DESC
                """)
                
                entity_counts = {}
                total_entities = 0
                async for record in entity_stats:
                    entity_type = record['entity_type']
                    count = record['count']
                    entity_counts[entity_type] = count
                    total_entities += count
                
                # Relationship counts by type
                rel_stats = await session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
                """)
                
                relationship_counts = {}
                total_relationships = 0
                async for record in rel_stats:
                    rel_type = record['relationship_type']
                    count = record['count']
                    relationship_counts[rel_type] = count
                    total_relationships += count
                
                # Document count
                doc_count = await session.run("MATCH (d:Document) RETURN count(d) as count")
                document_count = (await doc_count.single())['count']
                
                # Quality metrics
                quality_stats = await session.run("""
                MATCH (e:Entity)
                RETURN 
                    avg(e.quality_score) as avg_quality,
                    min(e.quality_score) as min_quality,
                    max(e.quality_score) as max_quality,
                    avg(e.confidence) as avg_confidence
                """)
                
                quality_record = await quality_stats.single()
                
                # Graph density
                density = 0.0
                if total_entities > 1:
                    max_possible_edges = total_entities * (total_entities - 1)
                    density = (total_relationships * 2) / max_possible_edges if max_possible_edges > 0 else 0.0
                
                return {
                    'total_entities': total_entities,
                    'total_relationships': total_relationships,
                    'total_documents': document_count,
                    'entity_counts': entity_counts,
                    'relationship_counts': relationship_counts,
                    'graph_density': density,
                    'quality_metrics': {
                        'average_quality_score': float(quality_record['avg_quality'] or 0.0),
                        'min_quality_score': float(quality_record['min_quality'] or 0.0),
                        'max_quality_score': float(quality_record['max_quality'] or 0.0),
                        'average_confidence': float(quality_record['avg_confidence'] or 0.0)
                    },
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {str(e)}")
            return {
                'total_entities': 0,
                'total_relationships': 0,
                'total_documents': 0,
                'entity_counts': {},
                'relationship_counts': {},
                'graph_density': 0.0,
                'quality_metrics': {},
                'last_updated': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def get_visualization_data(self, limit: int = 500) -> Dict[str, Any]:
        """Get graph data formatted for visualization"""
        try:
            async with self.driver.session() as session:
                # Get nodes (entities)
                nodes_result = await session.run("""
                MATCH (e:Entity)
                RETURN e
                ORDER BY e.quality_score DESC, e.mentions DESC
                LIMIT $limit
                """, limit=limit)
                
                nodes = []
                node_ids = set()
                
                async for record in nodes_result:
                    entity = dict(record['e'])
                    node_ids.add(entity['id'])
                    
                    # Parse attributes
                    if 'attributes' in entity:
                        try:
                            entity['attributes'] = json.loads(entity['attributes'])
                        except:
                            entity['attributes'] = {}
                    
                    # Format for visualization
                    node = {
                        'id': entity['id'],
                        'label': entity.get('name', entity['id']),
                        'type': entity.get('type', 'Entity'),
                        'size': min(max(entity.get('mentions', 1) * 5, 10), 50),
                        'color': self._get_node_color(entity.get('type', 'Entity')),
                        'quality_score': entity.get('quality_score', 0.0),
                        'confidence': entity.get('confidence', 0.0),
                        'attributes': entity.get('attributes', {})
                    }
                    nodes.append(node)
                
                # Get edges (relationships) between the selected nodes
                if node_ids:
                    edges_result = await session.run("""
                    MATCH (source:Entity)-[r]->(target:Entity)
                    WHERE source.id IN $node_ids AND target.id IN $node_ids
                    RETURN source.id as source, target.id as target, r
                    LIMIT $limit
                    """, node_ids=list(node_ids), limit=limit)
                    
                    edges = []
                    async for record in edges_result:
                        relationship = dict(record['r'])
                        
                        edge = {
                            'source': record['source'],
                            'target': record['target'],
                            'type': relationship.get('type', 'RELATED_TO'),
                            'strength': relationship.get('strength', 0.5),
                            'confidence': relationship.get('confidence', 0.0),
                            'quality_score': relationship.get('quality_score', 0.0),
                            'width': max(relationship.get('strength', 0.5) * 5, 1),
                            'color': self._get_edge_color(relationship.get('quality_score', 0.0))
                        }
                        edges.append(edge)
                else:
                    edges = []
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'metadata': {
                        'total_nodes': len(nodes),
                        'total_edges': len(edges),
                        'generated_at': datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get visualization data: {str(e)}")
            return {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'total_nodes': 0,
                    'total_edges': 0,
                    'generated_at': datetime.now().isoformat(),
                    'error': str(e)
                }
            }
    
    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map = {
            'Person': '#FF6B6B',
            'Place': '#4ECDC4',
            'Thing': '#45B7D1',
            'Concept': '#96CEB4',
            'Event': '#FFEAA7',
            'Organization': '#DDA0DD',
            'Entity': '#95A5A6'
        }
        return color_map.get(entity_type, '#95A5A6')
    
    def _get_edge_color(self, quality_score: float) -> str:
        """Get color for edge based on quality score"""
        if quality_score >= 0.8:
            return '#27AE60'  # Green for high quality
        elif quality_score >= 0.6:
            return '#F39C12'  # Orange for medium quality
        elif quality_score >= 0.4:
            return '#E67E22'  # Dark orange for low quality
        else:
            return '#E74C3C'  # Red for very low quality
    
    async def clear_graph(self) -> bool:
        """Clear all nodes and relationships from the graph"""
        try:
            async with self.driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
                logger.info("Graph cleared successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear graph: {str(e)}")
            return False
    
    async def update_entity_quality_score(self, entity_id: str, quality_score: float) -> bool:
        """Update the quality score of an entity"""
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            SET e.quality_score = $quality_score, e.updated_at = $timestamp
            RETURN e
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, 
                                         entity_id=entity_id, 
                                         quality_score=quality_score,
                                         timestamp=datetime.now().isoformat())
                record = await result.single()
                return record is not None
                
        except Exception as e:
            logger.error(f"Failed to update entity quality score: {str(e)}")
            return False
    
    async def merge_duplicate_entities(self, primary_id: str, duplicate_ids: List[str]) -> bool:
        """Merge duplicate entities into a primary entity"""
        try:
            async with self.driver.session() as session:
                # Start transaction
                async with session.begin_transaction() as tx:
                    # Get primary entity
                    primary_result = await tx.run("MATCH (p:Entity {id: $id}) RETURN p", id=primary_id)
                    primary_record = await primary_result.single()
                    
                    if not primary_record:
                        logger.error(f"Primary entity {primary_id} not found")
                        return False
                    
                    # Merge each duplicate
                    for dup_id in duplicate_ids:
                        # Transfer relationships
                        await tx.run("""
                        MATCH (dup:Entity {id: $dup_id})-[r]->(other)
                        MATCH (primary:Entity {id: $primary_id})
                        WHERE NOT (primary)-[]->(other)
                        CREATE (primary)-[new_r:RELATED_TO]->(other)
                        SET new_r = properties(r)
                        """, dup_id=dup_id, primary_id=primary_id)
                        
                        await tx.run("""
                        MATCH (other)-[r]->(dup:Entity {id: $dup_id})
                        MATCH (primary:Entity {id: $primary_id})
                        WHERE NOT (other)-[]->(primary)
                        CREATE (other)-[new_r:RELATED_TO]->(primary)
                        SET new_r = properties(r)
                        """, dup_id=dup_id, primary_id=primary_id)
                        
                        # Update mentions count
                        await tx.run("""
                        MATCH (dup:Entity {id: $dup_id})
                        MATCH (primary:Entity {id: $primary_id})
                        SET primary.mentions = primary.mentions + dup.mentions
                        """, dup_id=dup_id, primary_id=primary_id)
                        
                        # Delete duplicate
                        await tx.run("MATCH (dup:Entity {id: $dup_id}) DETACH DELETE dup", dup_id=dup_id)
                    
                    await tx.commit()
                    logger.info(f"Merged {len(duplicate_ids)} duplicate entities into {primary_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to merge duplicate entities: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Neo4j connection and database"""
        try:
            if not self.is_connected:
                return {
                    'status': 'disconnected',
                    'message': 'Not connected to Neo4j database',
                    'timestamp': datetime.now().isoformat()
                }
            
            async with self.driver.session() as session:
                # Test basic connectivity
                result = await session.run("RETURN 1 as test")
                await result.consume()
                
                # Get database info
                db_info = await session.run("CALL dbms.components() YIELD name, versions, edition")
                components = []
                async for record in db_info:
                    components.append({
                        'name': record['name'],
                        'versions': record['versions'],
                        'edition': record['edition']
                    })
                
                # Get basic statistics
                stats = await self.get_graph_statistics()
                
                return {
                    'status': 'healthy',
                    'message': 'Neo4j database is operational',
                    'database_info': components,
                    'graph_statistics': stats,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Neo4j health check failed: {str(e)}")
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
