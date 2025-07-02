# GraphRAG Setup Guide

This guide will help you set up and configure the enhanced GraphRAG (Graph Retrieval-Augmented Generation) system with Neo4j integration.

## Overview

The GraphRAG system combines traditional vector-based RAG with knowledge graph capabilities to provide enhanced contextual understanding and retrieval. It uses:

- **Neo4j** for graph database storage
- **CrewAI** for intelligent entity extraction workflows
- **Qwen LLM** for optimized entity recognition
- **ChromaDB** for vector embeddings
- **Interactive visualization** for graph quality verification

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- At least 8GB RAM (16GB recommended)
- GPU support (optional but recommended)

## Installation Steps

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Key new dependencies for GraphRAG:
# - neo4j>=5.15.0
# - networkx>=3.0
# - matplotlib>=3.7.0
# - plotly>=5.17.0
```

### 2. Start Neo4j Database

```bash
# Start Neo4j using Docker Compose
docker-compose up -d neo4j

# Verify Neo4j is running
docker logs graphrag_neo4j

# Access Neo4j Browser at http://localhost:7474
# Default credentials: neo4j/password
```

### 3. Configure Environment

Create or update your `.env` file:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
OLLAMA_URL=http://localhost:11434

# Qwen Model for Entity Extraction
QWEN_MODEL=qwen2.5:latest
```

### 4. Pull Required Models

```bash
# Pull Qwen model for entity extraction
ollama pull qwen2.5:latest

# Pull embedding model
ollama pull bge-m3:latest

# Pull other models as needed
ollama pull llama3.1:latest
```

## System Architecture

```
Document Upload → Entity Extraction (CrewAI + Qwen) → Graph Construction (Neo4j) → 
Quality Assessment → Iterative Refinement → Graph Visualization → 
Enhanced RAG → Synthetic Data Generation
```

### Key Components

1. **Neo4jManager**: Handles all graph database operations
2. **EntityExtractionCrew**: CrewAI workflow for entity extraction
3. **GraphRAGSystem**: Main system orchestrator
4. **CleanupManager**: Enhanced cleanup functionality

## Usage Guide

### 1. Start the System

```bash
# Start the backend server
python backend/app.py

# Or use the startup script
python start_server.py
```

### 2. Connect to GraphRAG

The system will automatically attempt to connect to Neo4j on startup. You can also manually connect:

```bash
curl -X POST http://localhost:8000/api/graphrag/connect
```

### 3. Upload and Process Documents

1. Upload documents through the web interface
2. Process documents through GraphRAG pipeline:

```bash
curl -X POST http://localhost:8000/api/graphrag/process-document/{document_id}
```

### 4. Query the System

```bash
curl -X POST http://localhost:8000/api/graphrag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main concepts in the document?",
    "max_results": 5,
    "use_graph_expansion": true,
    "graph_depth": 2
  }'
```

### 5. Visualize the Graph

Access the graph visualization through the troubleshooting interface:
- Navigate to the Troubleshooting section
- Click on "Graph Analysis" tab
- View interactive graph visualization

## Cleanup Operations

The system provides four types of cleanup operations:

### 1. Clear Queued Documents
- Removes only uploaded documents waiting for processing
- Preserves processed data and results

```bash
curl -X POST http://localhost:8000/api/cleanup/clear-queued-documents
```

### 2. Fresh Start Cleanup
- Complete system reset for new dataset generation
- Clears ALL data including documents, graph, vector DB, and results

```bash
curl -X POST http://localhost:8000/api/cleanup/fresh-start
```

### 3. Clear Graph Only
- Clears only the knowledge graph
- Preserves documents and results (graph can be rebuilt)

```bash
curl -X POST http://localhost:8000/api/cleanup/clear-graph-only
```

### 4. Optimize Graph
- Improves graph quality without full rebuild
- Merges duplicates, strengthens relationships, updates quality scores

```bash
curl -X POST http://localhost:8000/api/cleanup/optimize-graph
```

## Configuration Options

### Entity Extraction Settings

The system can be configured for different entity extraction strategies:

```python
# In backend/entity_extraction_crew.py
ENTITY_TYPES = [
    'Person',      # People, names, roles
    'Place',       # Locations, addresses
    'Organization', # Companies, institutions
    'Concept',     # Ideas, theories, methodologies
    'Event',       # Meetings, conferences, incidents
    'Thing'        # Objects, products, technologies
]

# Quality thresholds
MIN_ENTITY_CONFIDENCE = 0.5
MIN_RELATIONSHIP_STRENGTH = 0.4
MIN_GRAPH_QUALITY = 0.6
```

### Graph Visualization Settings

```python
# Node and edge limits for visualization
MAX_VISUALIZATION_NODES = 500
MAX_VISUALIZATION_EDGES = 1000

# Color schemes for different entity types
ENTITY_COLORS = {
    'Person': '#FF6B6B',
    'Place': '#4ECDC4',
    'Thing': '#45B7D1',
    'Concept': '#96CEB4',
    'Event': '#FFEAA7',
    'Organization': '#DDA0DD'
}
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   ```bash
   # Check if Neo4j is running
   docker ps | grep neo4j
   
   # Check logs
   docker logs graphrag_neo4j
   
   # Restart if needed
   docker-compose restart neo4j
   ```

2. **Entity Extraction Errors**
   ```bash
   # Verify Qwen model is available
   ollama list | grep qwen
   
   # Pull if missing
   ollama pull qwen2.5:latest
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Increase Docker memory limits if needed
   # Reduce batch sizes in processing
   ```

### Health Checks

```bash
# Overall system health
curl http://localhost:8000/api/graphrag/health

# Graph statistics
curl http://localhost:8000/api/graphrag/statistics

# Cleanup status
curl http://localhost:8000/api/cleanup/status
```

## Performance Optimization

### 1. Hardware Recommendations

- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ for large documents
- **GPU**: NVIDIA GPU with 8GB+ VRAM for faster processing
- **Storage**: SSD recommended for database performance

### 2. Configuration Tuning

```python
# Adjust batch sizes based on available memory
DOCUMENT_CHUNK_SIZE = 500  # Reduce if memory constrained
ENTITY_BATCH_SIZE = 100    # Process entities in batches
EMBEDDING_BATCH_SIZE = 50  # Embedding generation batch size
```

### 3. Neo4j Optimization

```cypher
// Create additional indexes for better performance
CREATE INDEX entity_quality_index IF NOT EXISTS FOR (e:Entity) ON (e.quality_score);
CREATE INDEX relationship_strength_index IF NOT EXISTS FOR ()-[r]-() ON (r.strength);
```

## Advanced Features

### 1. Custom Entity Types

You can extend the system with custom entity types:

```python
# Add to backend/entity_extraction_crew.py
CUSTOM_ENTITY_PATTERNS = {
    'Technology': [
        r'\b(?:AI|ML|blockchain|IoT|5G|cloud computing)\b',
        r'\b[A-Z][a-z]+ (?:API|SDK|framework|platform)\b'
    ],
    'Metric': [
        r'\b\d+(?:\.\d+)?%\b',
        r'\b\$\d+(?:,\d{3})*(?:\.\d{2})?\b'
    ]
}
```

### 2. Quality Scoring Customization

```python
# Customize quality scoring weights
QUALITY_WEIGHTS = {
    'entity_confidence': 0.25,
    'relationship_confidence': 0.25,
    'type_diversity': 0.15,
    'relationship_diversity': 0.15,
    'coverage': 0.10,
    'coherence': 0.10
}
```

### 3. Integration with External APIs

The system can be extended to integrate with external knowledge bases:

```python
# Example: Wikipedia API integration
async def enrich_entity_with_wikipedia(entity_name):
    # Fetch additional information from Wikipedia
    # Add to entity attributes
    pass
```

## Monitoring and Maintenance

### 1. Regular Maintenance Tasks

```bash
# Weekly: Optimize graph performance
curl -X POST http://localhost:8000/api/cleanup/optimize-graph

# Monthly: Full system health check
curl http://localhost:8000/api/graphrag/health

# As needed: Clear old results
curl -X DELETE http://localhost:8000/clear-results
```

### 2. Backup Procedures

```bash
# Backup Neo4j data
docker exec graphrag_neo4j neo4j-admin database dump neo4j

# Backup vector database
tar -czf vector_db_backup.tar.gz vector_db/

# Backup uploaded documents
tar -czf documents_backup.tar.gz uploads/
```

### 3. Monitoring Metrics

Key metrics to monitor:
- Graph density and quality scores
- Entity extraction accuracy
- Query response times
- Memory and CPU usage
- Database performance

## Security Considerations

1. **Database Security**
   - Change default Neo4j password
   - Use secure connections in production
   - Implement proper access controls

2. **API Security**
   - Add authentication for production use
   - Implement rate limiting
   - Validate all inputs

3. **Data Privacy**
   - Ensure sensitive data is properly handled
   - Implement data retention policies
   - Consider encryption for sensitive documents

## Support and Resources

- **Documentation**: See inline code documentation
- **Issues**: Report issues through the troubleshooting interface
- **Neo4j Documentation**: https://neo4j.com/docs/
- **CrewAI Documentation**: https://docs.crewai.com/
- **Graph Visualization**: Built-in D3.js visualization in troubleshooting interface

## Next Steps

1. Start with small documents to test the system
2. Gradually increase document size and complexity
3. Monitor performance and adjust settings as needed
4. Explore advanced features like custom entity types
5. Integrate with your existing workflows

The GraphRAG system provides a powerful foundation for enhanced document understanding and retrieval. With proper setup and configuration, it can significantly improve the quality and contextual awareness of your RAG applications.
