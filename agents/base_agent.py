"""
Base Agent class for the multi-agent data pipeline system.
All specialized agents inherit from this base class.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(Enum):
    """Defines the role of each agent in the system."""
    ORCHESTRATOR = "orchestrator"
    DATA_ANALYST = "data_analyst"
    ML_ENGINEER = "ml_engineer"
    VISUALIZATION = "visualization"
    INSIGHT = "insight"


@dataclass
class AgentState:
    """Shared state between agents."""
    data: Any = None  # DataFrame or processed data
    target_column: str = ""
    cleaning_report: str = ""
    model_results: Dict = field(default_factory=dict)
    visualizations: list = field(default_factory=list)
    insights: str = ""
    errors: list = field(default_factory=list)
    current_step: str = "initialized"


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    
    Features:
    - Lazy loading for memory efficiency
    - Error handling with graceful fallbacks
    - Logging for debugging
    - RAG integration capability
    """
    
    def __init__(
        self, 
        role: AgentRole,
        llm_api_key: Optional[str] = None,
        rag_retriever: Optional[Any] = None
    ):
        self.role = role
        self.llm_api_key = llm_api_key or os.environ.get("GROQ_API_KEY")
        self.rag_retriever = rag_retriever
        self._initialized = False
    
    def lazy_init(self):
        """Initialize heavy resources only when needed (memory optimization)."""
        if not self._initialized:
            self._setup()
            self._initialized = True
    
    def _setup(self):
        """Override in subclasses for custom initialization."""
        pass
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the agent's task.
        
        Args:
            state: Current shared state between agents
            
        Returns:
            Updated state after this agent's processing
        """
        pass
    
    def query_rag(self, query: str, k: int = 3) -> list:
        """
        Query the RAG system for relevant context.
        
        Args:
            query: The search query
            k: Number of results to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.rag_retriever is None:
            return []
        
        try:
            return self.rag_retriever.retrieve(query, k=k)
        except Exception as e:
            print(f"[{self.role.value}] RAG query failed: {e}")
            return []
    
    def call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call the LLM (Groq) for reasoning or generation.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response text
        """
        import requests
        
        if not self.llm_api_key:
            return "⚠️ LLM unavailable: No API key configured."
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "system", "content": f"You are a {self.role.value} agent in a data analysis pipeline. Be concise and actionable."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices and len(choices) > 0:
                return choices[0].get("message", {}).get("content", "No response")
            return "Empty response from LLM"
        except Exception as e:
            return f"⚠️ LLM call failed: {str(e)[:100]}"
    
    def log(self, message: str):
        """Log a message with agent context."""
        print(f"[{self.role.value.upper()}] {message}")
