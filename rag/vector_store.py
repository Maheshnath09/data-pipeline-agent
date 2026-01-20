"""
Vector Store using ChromaDB for the RAG system.
Optimized for memory-efficient operation on HF Spaces free tier.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class VectorStore:
    """
    ChromaDB-based vector store for RAG.
    
    Features:
    - Persistent storage (not in-memory) for memory efficiency
    - Lazy loading of embeddings model
    - Batch processing for large documents
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "data_pipeline_knowledge"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False
    
    def _lazy_init(self):
        """Initialize ChromaDB and embeddings only when needed."""
        if self._initialized:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persistent client for memory efficiency
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection with embedding function
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Data pipeline knowledge base"}
            )
            
            self._initialized = True
            print(f"[VectorStore] Initialized with {self._collection.count()} documents")
            
        except ImportError:
            print("[VectorStore] ChromaDB not installed. Run: pip install chromadb")
            raise
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers (lazy loaded)."""
        if self._embedding_fn is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, efficient model
                self._embedding_fn = SentenceTransformer('all-MiniLM-L6-v2')
                print("[VectorStore] Loaded embedding model: all-MiniLM-L6-v2")
            except ImportError:
                print("[VectorStore] sentence-transformers not installed")
                raise
        
        embeddings = self._embedding_fn.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text documents
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
            
        Returns:
            Number of documents added
        """
        self._lazy_init()
        
        if not documents:
            return 0
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self._collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self._get_embeddings(documents)
        
        # Add to collection
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )
        
        print(f"[VectorStore] Added {len(documents)} documents")
        return len(documents)
    
    def query(
        self,
        query_text: str,
        k: int = 3,
        where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: The query string
            k: Number of results to return
            where: Optional filter conditions
            
        Returns:
            List of matching documents with metadata and scores
        """
        self._lazy_init()
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query_text])[0]
        
        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0
                })
        
        return formatted_results
    
    def count(self) -> int:
        """Get the number of documents in the store."""
        self._lazy_init()
        return self._collection.count()
    
    def reset(self):
        """Clear all documents from the store."""
        self._lazy_init()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"description": "Data pipeline knowledge base"}
        )
        print("[VectorStore] Reset complete")
