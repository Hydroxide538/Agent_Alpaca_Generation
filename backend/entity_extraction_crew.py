"""
CrewAI Entity Extraction Workflow for GraphRAG
Specialized agents for entity extraction, relationship analysis, and quality assessment
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid
import asyncio
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import re

logger = logging.getLogger(__name__)

class EntityExtractionResult(BaseModel):
    """Result model for entity extraction"""
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: float = Field(default=0.0)
    confidence: float = Field(default=0.0)
    processing_time: float = Field(default=0.0)
    document_id: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EntityExtractionTool(BaseTool):
    """Tool for extracting entities from text using Qwen-optimized prompts"""
    
    name: str = "entity_extraction_tool"
    description: str = "Extract entities (people, places, things, concepts, events) from text with high accuracy"
    
    def _run(self, text: str, document_id: str = "") -> Dict[str, Any]:
        """Extract entities from text"""
        try:
            # Qwen-optimized entity extraction
            entities = self._extract_entities_qwen(text, document_id)
            return {
                "entities": entities,
                "success": True,
                "message": f"Extracted {len(entities)} entities"
            }
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return {
                "entities": [],
                "success": False,
                "message": f"Extraction failed: {str(e)}"
            }
    
    def _extract_entities_qwen(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Qwen-optimized entity extraction logic"""
        entities = []
        
        # Person extraction patterns (Qwen is good at NER)
        person_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Title First Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b'  # First M. Last
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group().strip()
                if len(name) > 3 and not self._is_common_word(name):
                    entities.append({
                        'id': str(uuid.uuid4()),
                        'name': name,
                        'type': 'Person',
                        'confidence': 0.8,
                        'source_document': document_id,
                        'context': text[max(0, match.start()-50):match.end()+50],
                        'position': match.start(),
                        'attributes': {
                            'extraction_method': 'pattern_matching',
                            'pattern_type': 'person'
                        }
                    })
        
        # Place extraction patterns
        place_patterns = [
            r'\b[A-Z][a-z]+ (?:City|County|State|Province|Country)\b',
            r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b',
            r'\b[A-Z][a-z]+ University\b',
            r'\b[A-Z][a-z]+ Hospital\b',
            r'\b[A-Z][a-z]+ Street\b',
            r'\b[A-Z][a-z]+ Avenue\b'
        ]
        
        for pattern in place_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group().strip()
                entities.append({
                    'id': str(uuid.uuid4()),
                    'name': name,
                    'type': 'Place',
                    'confidence': 0.7,
                    'source_document': document_id,
                    'context': text[max(0, match.start()-50):match.end()+50],
                    'position': match.start(),
                    'attributes': {
                        'extraction_method': 'pattern_matching',
                        'pattern_type': 'place'
                    }
                })
        
        # Organization extraction patterns
        org_patterns = [
            r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
            r'\b[A-Z][a-z]+ & [A-Z][a-z]+\b',
            r'\b(?:Apple|Google|Microsoft|Amazon|Facebook|Tesla|Netflix)\b'
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group().strip()
                entities.append({
                    'id': str(uuid.uuid4()),
                    'name': name,
                    'type': 'Organization',
                    'confidence': 0.75,
                    'source_document': document_id,
                    'context': text[max(0, match.start()-50):match.end()+50],
                    'position': match.start(),
                    'attributes': {
                        'extraction_method': 'pattern_matching',
                        'pattern_type': 'organization'
                    }
                })
        
        # Concept extraction (important terms, technical concepts)
        concept_patterns = [
            r'\b(?:artificial intelligence|machine learning|deep learning|neural network|algorithm)\b',
            r'\b(?:blockchain|cryptocurrency|bitcoin|ethereum)\b',
            r'\b(?:climate change|global warming|renewable energy|solar power)\b',
            r'\b[A-Z][a-z]+ (?:Theory|Principle|Law|Effect|Syndrome)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group().strip()
                entities.append({
                    'id': str(uuid.uuid4()),
                    'name': name,
                    'type': 'Concept',
                    'confidence': 0.6,
                    'source_document': document_id,
                    'context': text[max(0, match.start()-50):match.end()+50],
                    'position': match.start(),
                    'attributes': {
                        'extraction_method': 'pattern_matching',
                        'pattern_type': 'concept'
                    }
                })
        
        # Remove duplicates and sort by position
        unique_entities = {}
        for entity in entities:
            key = (entity['name'].lower(), entity['type'])
            if key not in unique_entities or entity['confidence'] > unique_entities[key]['confidence']:
                unique_entities[key] = entity
        
        return list(unique_entities.values())
    
    def _is_common_word(self, word: str) -> bool:
        """Check if word is a common word that shouldn't be an entity"""
        common_words = {
            'The Company', 'This Report', 'Last Year', 'Next Month', 
            'First Time', 'New York', 'United States', 'North America'
        }
        return word in common_words

class RelationshipAnalysisTool(BaseTool):
    """Tool for analyzing relationships between entities"""
    
    name: str = "relationship_analysis_tool"
    description: str = "Analyze and extract relationships between entities in text"
    
    def _run(self, entities: List[Dict[str, Any]], text: str, document_id: str = "") -> Dict[str, Any]:
        """Analyze relationships between entities"""
        try:
            relationships = self._extract_relationships(entities, text, document_id)
            return {
                "relationships": relationships,
                "success": True,
                "message": f"Found {len(relationships)} relationships"
            }
        except Exception as e:
            logger.error(f"Relationship analysis failed: {str(e)}")
            return {
                "relationships": [],
                "success": False,
                "message": f"Analysis failed: {str(e)}"
            }
    
    def _extract_relationships(self, entities: List[Dict[str, Any]], text: str, document_id: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Create entity position map
        entity_positions = {}
        for entity in entities:
            entity_positions[entity['name']] = {
                'id': entity['id'],
                'position': entity.get('position', 0),
                'type': entity['type']
            }
        
        # Relationship patterns
        relationship_patterns = [
            (r'(\w+(?:\s+\w+)*)\s+(?:works for|employed by|works at)\s+(\w+(?:\s+\w+)*)', 'WORKS_FOR'),
            (r'(\w+(?:\s+\w+)*)\s+(?:is the|serves as)\s+(?:CEO|president|director|manager)\s+of\s+(\w+(?:\s+\w+)*)', 'LEADS'),
            (r'(\w+(?:\s+\w+)*)\s+(?:founded|established|created)\s+(\w+(?:\s+\w+)*)', 'FOUNDED'),
            (r'(\w+(?:\s+\w+)*)\s+(?:located in|based in|situated in)\s+(\w+(?:\s+\w+)*)', 'LOCATED_IN'),
            (r'(\w+(?:\s+\w+)*)\s+(?:owns|acquired|purchased)\s+(\w+(?:\s+\w+)*)', 'OWNS'),
            (r'(\w+(?:\s+\w+)*)\s+(?:partnered with|collaborated with|worked with)\s+(\w+(?:\s+\w+)*)', 'PARTNERED_WITH'),
            (r'(\w+(?:\s+\w+)*)\s+(?:and|&)\s+(\w+(?:\s+\w+)*)', 'ASSOCIATED_WITH')
        ]
        
        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_name = match.group(1).strip()
                target_name = match.group(2).strip()
                
                # Find matching entities
                source_entity = None
                target_entity = None
                
                for entity in entities:
                    if entity['name'].lower() == source_name.lower():
                        source_entity = entity
                    if entity['name'].lower() == target_name.lower():
                        target_entity = entity
                
                if source_entity and target_entity and source_entity['id'] != target_entity['id']:
                    relationships.append({
                        'id': str(uuid.uuid4()),
                        'source_id': source_entity['id'],
                        'target_id': target_entity['id'],
                        'type': rel_type,
                        'confidence': 0.7,
                        'strength': 0.8,
                        'source_document': document_id,
                        'context': match.group(0),
                        'position': match.start(),
                        'attributes': {
                            'extraction_method': 'pattern_matching',
                            'source_type': source_entity['type'],
                            'target_type': target_entity['type']
                        }
                    })
        
        # Proximity-based relationships (entities mentioned close together)
        sorted_entities = sorted(entities, key=lambda x: x.get('position', 0))
        
        for i, entity1 in enumerate(sorted_entities):
            for j, entity2 in enumerate(sorted_entities[i+1:], i+1):
                pos1 = entity1.get('position', 0)
                pos2 = entity2.get('position', 0)
                
                # If entities are within 100 characters, create a weak relationship
                if abs(pos2 - pos1) < 100:
                    relationships.append({
                        'id': str(uuid.uuid4()),
                        'source_id': entity1['id'],
                        'target_id': entity2['id'],
                        'type': 'CO_OCCURS',
                        'confidence': 0.4,
                        'strength': 0.3,
                        'source_document': document_id,
                        'context': f"Entities mentioned within {abs(pos2 - pos1)} characters",
                        'position': min(pos1, pos2),
                        'attributes': {
                            'extraction_method': 'proximity',
                            'distance': abs(pos2 - pos1),
                            'source_type': entity1['type'],
                            'target_type': entity2['type']
                        }
                    })
        
        return relationships

class QualityAssessmentTool(BaseTool):
    """Tool for assessing the quality of entity extraction and relationships"""
    
    name: str = "quality_assessment_tool"
    description: str = "Assess the quality of extracted entities and relationships"
    
    def _run(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """Assess extraction quality"""
        try:
            quality_metrics = self._calculate_quality_metrics(entities, relationships, text)
            return {
                "quality_metrics": quality_metrics,
                "success": True,
                "message": "Quality assessment completed"
            }
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {
                "quality_metrics": {},
                "success": False,
                "message": f"Assessment failed: {str(e)}"
            }
    
    def _calculate_quality_metrics(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        
        # Entity quality metrics
        entity_confidence_avg = sum(e.get('confidence', 0) for e in entities) / len(entities) if entities else 0
        entity_type_diversity = len(set(e.get('type', 'Unknown') for e in entities))
        entity_density = len(entities) / len(text.split()) if text else 0
        
        # Relationship quality metrics
        rel_confidence_avg = sum(r.get('confidence', 0) for r in relationships) / len(relationships) if relationships else 0
        rel_type_diversity = len(set(r.get('type', 'Unknown') for r in relationships))
        rel_to_entity_ratio = len(relationships) / len(entities) if entities else 0
        
        # Coverage metrics
        entity_coverage = self._calculate_entity_coverage(entities, text)
        relationship_coverage = self._calculate_relationship_coverage(relationships, entities)
        
        # Overall quality score
        quality_score = (
            entity_confidence_avg * 0.25 +
            rel_confidence_avg * 0.25 +
            min(entity_type_diversity / 5, 1.0) * 0.15 +
            min(rel_type_diversity / 8, 1.0) * 0.15 +
            min(entity_coverage, 1.0) * 0.1 +
            min(relationship_coverage, 1.0) * 0.1
        )
        
        return {
            'overall_quality_score': quality_score,
            'entity_metrics': {
                'count': len(entities),
                'average_confidence': entity_confidence_avg,
                'type_diversity': entity_type_diversity,
                'density': entity_density,
                'coverage': entity_coverage
            },
            'relationship_metrics': {
                'count': len(relationships),
                'average_confidence': rel_confidence_avg,
                'type_diversity': rel_type_diversity,
                'entity_ratio': rel_to_entity_ratio,
                'coverage': relationship_coverage
            },
            'recommendations': self._generate_recommendations(entities, relationships, quality_score)
        }
    
    def _calculate_entity_coverage(self, entities: List[Dict[str, Any]], text: str) -> float:
        """Calculate how well entities cover the important parts of the text"""
        if not entities or not text:
            return 0.0
        
        # Simple heuristic: ratio of unique entity characters to total text length
        entity_chars = set()
        for entity in entities:
            for char in entity.get('name', '').lower():
                if char.isalnum():
                    entity_chars.add(char)
        
        text_chars = set()
        for char in text.lower():
            if char.isalnum():
                text_chars.add(char)
        
        if not text_chars:
            return 0.0
        
        return len(entity_chars) / len(text_chars)
    
    def _calculate_relationship_coverage(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> float:
        """Calculate how well relationships connect the entities"""
        if not relationships or not entities:
            return 0.0
        
        # Calculate the ratio of connected entities
        connected_entities = set()
        for rel in relationships:
            connected_entities.add(rel.get('source_id'))
            connected_entities.add(rel.get('target_id'))
        
        return len(connected_entities) / len(entities) if entities else 0.0
    
    def _generate_recommendations(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], quality_score: float) -> List[str]:
        """Generate recommendations for improving extraction quality"""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.append("Overall quality is low. Consider re-processing with different parameters.")
        
        if len(entities) < 5:
            recommendations.append("Few entities extracted. Text might need preprocessing or different extraction methods.")
        
        if len(relationships) == 0:
            recommendations.append("No relationships found. Consider using more sophisticated relationship extraction.")
        
        entity_types = set(e.get('type', 'Unknown') for e in entities)
        if len(entity_types) < 3:
            recommendations.append("Low entity type diversity. Consider expanding entity type recognition.")
        
        avg_confidence = sum(e.get('confidence', 0) for e in entities) / len(entities) if entities else 0
        if avg_confidence < 0.6:
            recommendations.append("Low average entity confidence. Consider manual review and validation.")
        
        return recommendations

class EntityExtractionCrew:
    """CrewAI workflow for entity extraction and graph construction"""
    
    def __init__(self, llm_manager, neo4j_manager):
        self.llm_manager = llm_manager
        self.neo4j_manager = neo4j_manager
        self.tools = {
            'entity_extraction': EntityExtractionTool(),
            'relationship_analysis': RelationshipAnalysisTool(),
            'quality_assessment': QualityAssessmentTool()
        }
        
    def create_agents(self, manager_llm, qwen_llm) -> Dict[str, Agent]:
        """Create specialized agents for entity extraction workflow"""
        
        # Entity Extraction Agent (uses Qwen for optimal performance)
        entity_extractor = Agent(
            role="Entity Extraction Specialist",
            goal="Extract high-quality entities (people, places, things, concepts, events) from documents with maximum accuracy",
            backstory="""You are an expert in named entity recognition with deep knowledge of linguistic patterns 
            and entity types. You specialize in using Qwen models for optimal entity extraction performance. 
            You understand the nuances of different entity types and can distinguish between similar entities.""",
            tools=[self.tools['entity_extraction']],
            llm=qwen_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        # Relationship Analysis Agent
        relationship_analyzer = Agent(
            role="Relationship Analysis Expert",
            goal="Identify and analyze relationships between entities with high precision and contextual understanding",
            backstory="""You are a specialist in relationship extraction and graph analysis. You understand 
            complex relationships between entities and can identify both explicit and implicit connections. 
            You excel at determining relationship strength and confidence levels.""",
            tools=[self.tools['relationship_analysis']],
            llm=manager_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        # Quality Assessment Agent
        quality_assessor = Agent(
            role="Quality Assessment Analyst",
            goal="Evaluate the quality of entity extraction and relationships, providing actionable improvement recommendations",
            backstory="""You are a quality assurance expert specializing in knowledge graph construction. 
            You have deep understanding of what makes high-quality entity extraction and relationship analysis. 
            You provide detailed metrics and actionable recommendations for improvement.""",
            tools=[self.tools['quality_assessment']],
            llm=manager_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
        
        # Graph Refinement Agent
        graph_refiner = Agent(
            role="Graph Refinement Specialist",
            goal="Iteratively improve the knowledge graph by identifying and fixing quality issues",
            backstory="""You are an expert in knowledge graph optimization and refinement. You can identify 
            duplicate entities, weak relationships, and areas for improvement. You work to enhance the overall 
            quality and coherence of the knowledge graph.""",
            llm=manager_llm,
            verbose=True,
            allow_delegation=True,
            max_iter=2
        )
        
        return {
            'entity_extractor': entity_extractor,
            'relationship_analyzer': relationship_analyzer,
            'quality_assessor': quality_assessor,
            'graph_refiner': graph_refiner
        }
    
    def create_tasks(self, agents: Dict[str, Agent], document_text: str, document_id: str) -> List[Task]:
        """Create tasks for the entity extraction workflow"""
        
        # Task 1: Extract entities
        extract_entities_task = Task(
            description=f"""
            Extract all relevant entities from the following document text. Focus on:
            1. People (names, titles, roles)
            2. Places (locations, addresses, geographical entities)
            3. Things (objects, products, technologies)
            4. Concepts (ideas, theories, methodologies)
            5. Events (meetings, conferences, incidents)
            6. Organizations (companies, institutions, groups)
            
            Document ID: {document_id}
            Document Text: {document_text[:2000]}...
            
            Use the entity_extraction_tool to perform the extraction. Ensure high accuracy and confidence scores.
            """,
            agent=agents['entity_extractor'],
            expected_output="A comprehensive list of extracted entities with confidence scores and metadata"
        )
        
        # Task 2: Analyze relationships
        analyze_relationships_task = Task(
            description=f"""
            Analyze the extracted entities and identify relationships between them. Look for:
            1. Direct relationships (works for, located in, owns, etc.)
            2. Implicit relationships (co-occurrence, contextual connections)
            3. Hierarchical relationships (part of, member of, etc.)
            4. Temporal relationships (before, after, during)
            
            Use the relationship_analysis_tool with the entities from the previous task.
            Provide confidence scores and relationship strength for each connection.
            """,
            agent=agents['relationship_analyzer'],
            expected_output="A detailed analysis of relationships between entities with confidence and strength metrics"
        )
        
        # Task 3: Assess quality
        assess_quality_task = Task(
            description=f"""
            Evaluate the quality of the entity extraction and relationship analysis. Provide:
            1. Overall quality score (0-1)
            2. Entity quality metrics (confidence, coverage, diversity)
            3. Relationship quality metrics (accuracy, completeness, coherence)
            4. Specific recommendations for improvement
            5. Areas that need manual review or re-processing
            
            Use the quality_assessment_tool with the results from previous tasks.
            """,
            agent=agents['quality_assessor'],
            expected_output="Comprehensive quality assessment with metrics and improvement recommendations"
        )
        
        # Task 4: Refine and optimize
        refine_graph_task = Task(
            description=f"""
            Based on the quality assessment, refine and optimize the extracted entities and relationships:
            1. Identify and merge duplicate entities
            2. Strengthen weak relationships with additional evidence
            3. Remove low-confidence entities and relationships if necessary
            4. Enhance entity attributes and relationship context
            5. Provide final quality score and validation status
            
            Work with the quality assessor to ensure improvements meet quality standards.
            """,
            agent=agents['graph_refiner'],
            expected_output="Refined and optimized entity-relationship graph with improved quality metrics"
        )
        
        return [extract_entities_task, analyze_relationships_task, assess_quality_task, refine_graph_task]
    
    async def process_document(self, document_text: str, document_id: str, manager_llm, qwen_llm) -> EntityExtractionResult:
        """Process a document through the entity extraction workflow"""
        try:
            start_time = datetime.now()
            
            # Create agents and tasks
            agents = self.create_agents(manager_llm, qwen_llm)
            tasks = self.create_tasks(agents, document_text, document_id)
            
            # Create and run crew
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
                manager_llm=manager_llm
            )
            
            # Execute workflow
            logger.info(f"Starting entity extraction workflow for document {document_id}")
            result = crew.kickoff()
            
            # Parse results (this would need to be adapted based on actual CrewAI output format)
            entities, relationships, quality_metrics = self._parse_crew_results(result)
            
            # Store in Neo4j
            await self._store_in_neo4j(entities, relationships, document_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EntityExtractionResult(
                entities=entities,
                relationships=relationships,
                quality_score=quality_metrics.get('overall_quality_score', 0.0),
                confidence=quality_metrics.get('entity_metrics', {}).get('average_confidence', 0.0),
                processing_time=processing_time,
                document_id=document_id,
                metadata={
                    'quality_metrics': quality_metrics,
                    'agent_count': len(agents),
                    'task_count': len(tasks),
                    'processed_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Entity extraction workflow failed: {str(e)}")
            return EntityExtractionResult(
                entities=[],
                relationships=[],
                quality_score=0.0,
                confidence=0.0,
                processing_time=0.0,
                document_id=document_id,
                metadata={'error': str(e)}
            )
    
    def _parse_crew_results(self, crew_result) -> tuple:
        """Parse CrewAI results into entities, relationships, and quality metrics"""
        # This is a placeholder - actual implementation would depend on CrewAI output format
        entities = []
        relationships = []
        quality_metrics = {}
        
        # Extract information from crew result
        # This would need to be implemented based on the actual CrewAI output structure
        
        return entities, relationships, quality_metrics
    
    async def _store_in_neo4j(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], document_id: str):
        """Store extracted entities and relationships in Neo4j"""
        try:
            if not self.neo4j_manager.is_connected:
                await self.neo4j_manager.connect()
            
            # Create document node
            await self.neo4j_manager.create_document_node({
                'id': document_id,
                'name': f"Document_{document_id}",
                'processed_at': datetime.now().isoformat()
            })
            
            # Create entity nodes
            entity_id_map = {}
            for entity in entities:
                entity_id = await self.neo4j_manager.create_entity(entity)
                entity_id_map[entity['id']] = entity_id
                
                # Link entity to document
                await self.neo4j_manager.link_entity_to_document(
                    entity_id, document_id, entity.get('context', '')
                )
            
            # Create relationships
            for relationship in relationships:
                source_id = entity_id_map.get(relationship['source_id'])
                target_id = entity_id_map.get(relationship['target_id'])
                
                if source_id and target_id:
                    await self.neo4j_manager.create_relationship(
                        source_id, target_id, relationship
                    )
            
            logger.info(f"Stored {len(entities)} entities and {len(relationships)} relationships in Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to store results in Neo4j: {str(e)}")
            raise
