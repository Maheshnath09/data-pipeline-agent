"""
RAG Retriever - Handles document retrieval and context augmentation.
"""
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever.
    
    Features:
    - Retrieves relevant context from vector store
    - Formats context for LLM prompts
    - Handles fallbacks gracefully
    """
    
    def __init__(
        self, 
        vector_store: Optional[VectorStore] = None,
        default_k: int = 3
    ):
        self.vector_store = vector_store or VectorStore()
        self.default_k = default_k
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            k: Number of results (default: self.default_k)
            category: Optional category filter
            
        Returns:
            List of relevant documents with metadata
        """
        k = k or self.default_k
        
        # Build filter if category specified
        where_filter = None
        if category:
            where_filter = {"category": category}
        
        try:
            results = self.vector_store.query(
                query_text=query,
                k=k,
                where=where_filter
            )
            return results
        except Exception as e:
            print(f"[RAGRetriever] Retrieval failed: {e}")
            return []
    
    def get_context(
        self,
        query: str,
        k: Optional[int] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Get formatted context string for LLM prompts.
        
        Args:
            query: The search query
            k: Number of results
            category: Optional category filter
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k, category)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            content = doc.get('content', '')
            category = doc.get('metadata', {}).get('category', 'general')
            context_parts.append(f"[{i}] ({category}) {content}")
        
        return "\n\n".join(context_parts)
    
    def augment_prompt(
        self,
        prompt: str,
        context_query: Optional[str] = None,
        k: int = 3
    ) -> str:
        """
        Augment a prompt with relevant context from RAG.
        
        Args:
            prompt: The original prompt
            context_query: Query to search for context (uses prompt if None)
            k: Number of context documents
            
        Returns:
            Augmented prompt with context
        """
        query = context_query or prompt
        context = self.get_context(query, k=k)
        
        if context == "No relevant context found.":
            return prompt
        
        augmented = f"""Use the following context to help answer the question:

### Context:
{context}

### Question:
{prompt}

### Answer:"""
        
        return augmented
