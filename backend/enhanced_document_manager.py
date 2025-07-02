import os
import json
import logging
import uuid
import shutil
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import mimetypes

from backend.token_counter import TokenCounter

logger = logging.getLogger(__name__)

class DocumentCollection:
    """Represents a collection of related documents"""
    
    def __init__(self, collection_id: str, name: str, description: str = ""):
        self.collection_id = collection_id
        self.name = name
        self.description = description
        self.documents = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.metadata = {}
    
    def add_document(self, document: Dict[str, Any]):
        """Add a document to the collection"""
        self.documents.append(document)
        self.updated_at = datetime.now().isoformat()
    
    def remove_document(self, document_id: str):
        """Remove a document from the collection"""
        self.documents = [doc for doc in self.documents if doc.get("id") != document_id]
        self.updated_at = datetime.now().isoformat()
    
    def get_total_tokens(self) -> int:
        """Get total token count for all documents in collection"""
        return sum(doc.get("token_count", 0) for doc in self.documents)
    
    def get_total_size(self) -> int:
        """Get total file size for all documents in collection"""
        return sum(doc.get("size", 0) for doc in self.documents)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary"""
        return {
            "collection_id": self.collection_id,
            "name": self.name,
            "description": self.description,
            "documents": self.documents,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "stats": {
                "document_count": len(self.documents),
                "total_tokens": self.get_total_tokens(),
                "total_size": self.get_total_size()
            }
        }

class EnhancedDocumentManager:
    """Enhanced document manager with collection support and directory uploads"""
    
    def __init__(self, upload_dir: str = "uploads", collections_dir: str = "collections"):
        self.upload_dir = upload_dir
        self.collections_dir = collections_dir
        self.token_counter = TokenCounter()
        
        # Ensure directories exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.collections_dir, exist_ok=True)
        
        # Supported file types
        self.supported_extensions = {'.txt', '.csv', '.pdf', '.md', '.json', '.xml'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB per file
    
    async def upload_files(self, files: List[Any], collection_name: str = None) -> Dict[str, Any]:
        """Upload multiple files and optionally add to a collection"""
        try:
            uploaded_documents = []
            total_tokens = 0
            
            # Create collection if specified
            collection = None
            if collection_name:
                collection = await self.create_collection(collection_name)
            
            for file in files:
                # Validate file
                validation_result = self._validate_file(file)
                if not validation_result["valid"]:
                    logger.warning(f"Skipping invalid file {file.filename}: {validation_result['reason']}")
                    continue
                
                # Process file
                document = await self._process_single_file(file)
                if document:
                    uploaded_documents.append(document)
                    total_tokens += document.get("token_count", 0)
                    
                    # Add to collection if specified
                    if collection:
                        collection.add_document(document)
            
            # Save collection if created
            if collection:
                await self._save_collection(collection)
            
            return {
                "success": True,
                "documents": uploaded_documents,
                "count": len(uploaded_documents),
                "total_tokens": total_tokens,
                "collection": collection.to_dict() if collection else None,
                "token_summary": {
                    "total_tokens": total_tokens,
                    "total_characters": sum(doc.get("character_count", 0) for doc in uploaded_documents),
                    "total_words": sum(doc.get("word_count", 0) for doc in uploaded_documents),
                    "encoding": self.token_counter.encoding_name
                }
            }
            
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise
    
    async def upload_directory(self, directory_path: str, collection_name: str = None, 
                             recursive: bool = True, file_filter: str = None) -> Dict[str, Any]:
        """Upload all files from a directory"""
        try:
            if not os.path.exists(directory_path):
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            # Find all files in directory
            file_paths = []
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._should_include_file(file_path, file_filter):
                            file_paths.append(file_path)
            else:
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)
                    if os.path.isfile(item_path) and self._should_include_file(item_path, file_filter):
                        file_paths.append(item_path)
            
            if not file_paths:
                return {
                    "success": True,
                    "documents": [],
                    "count": 0,
                    "total_tokens": 0,
                    "message": "No supported files found in directory"
                }
            
            # Create collection
            collection = None
            if collection_name:
                collection = await self.create_collection(collection_name, 
                                                        f"Directory upload from {directory_path}")
            
            # Process files
            uploaded_documents = []
            total_tokens = 0
            
            for file_path in file_paths:
                try:
                    document = await self._process_file_from_path(file_path)
                    if document:
                        uploaded_documents.append(document)
                        total_tokens += document.get("token_count", 0)
                        
                        # Add to collection
                        if collection:
                            collection.add_document(document)
                            
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {str(e)}")
                    continue
            
            # Save collection
            if collection:
                await self._save_collection(collection)
            
            return {
                "success": True,
                "documents": uploaded_documents,
                "count": len(uploaded_documents),
                "total_tokens": total_tokens,
                "collection": collection.to_dict() if collection else None,
                "source_directory": directory_path,
                "processed_files": len(file_paths),
                "token_summary": {
                    "total_tokens": total_tokens,
                    "total_characters": sum(doc.get("character_count", 0) for doc in uploaded_documents),
                    "total_words": sum(doc.get("word_count", 0) for doc in uploaded_documents),
                    "encoding": self.token_counter.encoding_name
                }
            }
            
        except Exception as e:
            logger.error(f"Directory upload failed: {str(e)}")
            raise
    
    async def create_collection(self, name: str, description: str = "") -> DocumentCollection:
        """Create a new document collection"""
        collection_id = str(uuid.uuid4())
        collection = DocumentCollection(collection_id, name, description)
        return collection
    
    async def get_collections(self) -> List[Dict[str, Any]]:
        """Get all document collections"""
        try:
            collections = []
            
            if not os.path.exists(self.collections_dir):
                return collections
            
            for filename in os.listdir(self.collections_dir):
                if filename.endswith('.json'):
                    collection_path = os.path.join(self.collections_dir, filename)
                    try:
                        with open(collection_path, 'r', encoding='utf-8') as f:
                            collection_data = json.load(f)
                            collections.append(collection_data)
                    except Exception as e:
                        logger.error(f"Failed to load collection {filename}: {str(e)}")
                        continue
            
            # Sort by updated_at (newest first)
            collections.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return collections
            
        except Exception as e:
            logger.error(f"Failed to get collections: {str(e)}")
            return []
    
    async def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific collection by ID"""
        try:
            collection_path = os.path.join(self.collections_dir, f"{collection_id}.json")
            if os.path.exists(collection_path):
                with open(collection_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to get collection {collection_id}: {str(e)}")
            return None
    
    async def delete_collection(self, collection_id: str, delete_files: bool = False) -> bool:
        """Delete a collection and optionally its files"""
        try:
            collection = await self.get_collection(collection_id)
            if not collection:
                return False
            
            # Delete files if requested
            if delete_files:
                for document in collection.get("documents", []):
                    file_path = document.get("path")
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.warning(f"Failed to delete file {file_path}: {str(e)}")
            
            # Delete collection file
            collection_path = os.path.join(self.collections_dir, f"{collection_id}.json")
            if os.path.exists(collection_path):
                os.remove(collection_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_id}: {str(e)}")
            return False
    
    async def add_documents_to_collection(self, collection_id: str, document_ids: List[str]) -> bool:
        """Add existing documents to a collection"""
        try:
            collection_data = await self.get_collection(collection_id)
            if not collection_data:
                return False
            
            # Get documents by ID
            for doc_id in document_ids:
                document = await self.get_document_by_id(doc_id)
                if document:
                    # Check if document is already in collection
                    existing_ids = [doc.get("id") for doc in collection_data.get("documents", [])]
                    if doc_id not in existing_ids:
                        collection_data["documents"].append(document)
            
            # Update collection
            collection_data["updated_at"] = datetime.now().isoformat()
            
            # Recalculate stats
            collection_data["stats"] = {
                "document_count": len(collection_data["documents"]),
                "total_tokens": sum(doc.get("token_count", 0) for doc in collection_data["documents"]),
                "total_size": sum(doc.get("size", 0) for doc in collection_data["documents"])
            }
            
            # Save collection
            collection_path = os.path.join(self.collections_dir, f"{collection_id}.json")
            with open(collection_path, 'w', encoding='utf-8') as f:
                json.dump(collection_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to collection {collection_id}: {str(e)}")
            return False
    
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all uploaded documents"""
        try:
            documents = []
            
            if not os.path.exists(self.upload_dir):
                return documents
            
            # Scan upload directory for document metadata
            for filename in os.listdir(self.upload_dir):
                if filename.endswith('.json') and filename.startswith('doc_'):
                    metadata_path = os.path.join(self.upload_dir, filename)
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            document_data = json.load(f)
                            documents.append(document_data)
                    except Exception as e:
                        logger.error(f"Failed to load document metadata {filename}: {str(e)}")
                        continue
            
            # Sort by upload time (newest first)
            documents.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {str(e)}")
            return []
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            metadata_path = os.path.join(self.upload_dir, f"doc_{document_id}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its metadata"""
        try:
            document = await self.get_document_by_id(document_id)
            if not document:
                return False
            
            # Delete the actual file
            file_path = document.get("path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete metadata
            metadata_path = os.path.join(self.upload_dir, f"doc_{document_id}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def clear_all_documents(self) -> Dict[str, Any]:
        """Clear all documents and collections"""
        try:
            deleted_files = 0
            deleted_collections = 0
            
            # Clear upload directory
            if os.path.exists(self.upload_dir):
                for filename in os.listdir(self.upload_dir):
                    file_path = os.path.join(self.upload_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_files += 1
            
            # Clear collections directory
            if os.path.exists(self.collections_dir):
                for filename in os.listdir(self.collections_dir):
                    if filename.endswith('.json'):
                        collection_path = os.path.join(self.collections_dir, filename)
                        os.remove(collection_path)
                        deleted_collections += 1
            
            return {
                "success": True,
                "deleted_files": deleted_files,
                "deleted_collections": deleted_collections,
                "message": f"Cleared {deleted_files} files and {deleted_collections} collections"
            }
            
        except Exception as e:
            logger.error(f"Failed to clear documents: {str(e)}")
            raise
    
    def _validate_file(self, file) -> Dict[str, Any]:
        """Validate uploaded file"""
        try:
            # Check file extension
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in self.supported_extensions:
                return {
                    "valid": False,
                    "reason": f"Unsupported file type: {file_extension}"
                }
            
            # Check file size (if available)
            if hasattr(file, 'size') and file.size > self.max_file_size:
                return {
                    "valid": False,
                    "reason": f"File too large: {file.size} bytes (max: {self.max_file_size})"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    def _should_include_file(self, file_path: str, file_filter: str = None) -> bool:
        """Check if file should be included based on extension and filter"""
        try:
            # Check extension
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_extensions:
                return False
            
            # Check file filter (glob pattern)
            if file_filter:
                import fnmatch
                filename = os.path.basename(file_path)
                if not fnmatch.fnmatch(filename, file_filter):
                    return False
            
            # Check file size
            if os.path.getsize(file_path) > self.max_file_size:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {str(e)}")
            return False
    
    async def _process_single_file(self, file) -> Optional[Dict[str, Any]]:
        """Process a single uploaded file"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix
            unique_filename = f"{file_id}{file_extension}"
            file_path = os.path.join(self.upload_dir, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Count tokens
            token_stats = self.token_counter.count_tokens_in_file(file_path)
            
            # Create document metadata
            document = {
                "id": file_id,
                "original_name": file.filename,
                "filename": unique_filename,
                "path": file_path,
                "size": os.path.getsize(file_path),
                "type": file_extension[1:].upper() if file_extension else "UNKNOWN",
                "upload_time": datetime.now().isoformat(),
                "token_count": token_stats.get("token_count", 0),
                "character_count": token_stats.get("character_count", 0),
                "word_count": token_stats.get("word_count", 0),
                "encoding": token_stats.get("encoding", "unknown"),
                "mime_type": mimetypes.guess_type(file.filename)[0]
            }
            
            # Save document metadata
            metadata_path = os.path.join(self.upload_dir, f"doc_{file_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {str(e)}")
            return None
    
    async def _process_file_from_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a file from a file system path"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            original_filename = os.path.basename(file_path)
            file_extension = Path(file_path).suffix
            unique_filename = f"{file_id}{file_extension}"
            destination_path = os.path.join(self.upload_dir, unique_filename)
            
            # Copy file to upload directory
            shutil.copy2(file_path, destination_path)
            
            # Count tokens
            token_stats = self.token_counter.count_tokens_in_file(destination_path)
            
            # Create document metadata
            document = {
                "id": file_id,
                "original_name": original_filename,
                "filename": unique_filename,
                "path": destination_path,
                "source_path": file_path,
                "size": os.path.getsize(destination_path),
                "type": file_extension[1:].upper() if file_extension else "UNKNOWN",
                "upload_time": datetime.now().isoformat(),
                "token_count": token_stats.get("token_count", 0),
                "character_count": token_stats.get("character_count", 0),
                "word_count": token_stats.get("word_count", 0),
                "encoding": token_stats.get("encoding", "unknown"),
                "mime_type": mimetypes.guess_type(original_filename)[0]
            }
            
            # Save document metadata
            metadata_path = os.path.join(self.upload_dir, f"doc_{file_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process file from path {file_path}: {str(e)}")
            return None
    
    async def _save_collection(self, collection: DocumentCollection):
        """Save collection to disk"""
        try:
            collection_path = os.path.join(self.collections_dir, f"{collection.collection_id}.json")
            with open(collection_path, 'w', encoding='utf-8') as f:
                json.dump(collection.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save collection {collection.collection_id}: {str(e)}")
            raise
