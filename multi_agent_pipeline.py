"""
Multi-Agent Pipeline - The main entry point for the multi-agent system.
Orchestrates all agents and provides a simple interface for the Gradio UI.
"""
import os
import pandas as pd
from typing import Optional, Tuple, Callable
from pathlib import Path

from agents import (
    AgentState,
    OrchestratorAgent,
    DataAnalystAgent,
    MLEngineerAgent,
    VisualizationAgent,
    InsightAgent
)
from rag import RAGRetriever, VectorStore


class MultiAgentPipeline:
    """
    Multi-Agent Pipeline for data analysis.
    
    Features:
    - Orchestrates 4 specialized agents
    - RAG-powered context retrieval
    - Progress tracking for UI
    - Production-grade error handling
    """
    
    def __init__(self, initialize_rag: bool = True):
        """
        Initialize the multi-agent pipeline.
        
        Args:
            initialize_rag: Whether to initialize RAG (set False for faster startup)
        """
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.rag_retriever = None
        self.orchestrator = None
        self._initialized = False
        self._initialize_rag = initialize_rag
    
    def _lazy_init(self):
        """Lazy initialization for memory efficiency."""
        if self._initialized:
            return
        
        print("[MultiAgentPipeline] Initializing...")
        
        # Initialize RAG if enabled
        if self._initialize_rag:
            try:
                persist_dir = Path(__file__).parent / "chroma_db"
                vector_store = VectorStore(persist_directory=str(persist_dir))
                self.rag_retriever = RAGRetriever(vector_store=vector_store)
                print("[MultiAgentPipeline] RAG initialized")
            except Exception as e:
                print(f"[MultiAgentPipeline] RAG initialization failed: {e}")
                self.rag_retriever = None
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent(
            llm_api_key=self.api_key,
            rag_retriever=self.rag_retriever
        )
        
        # Register specialized agents
        agents = [
            DataAnalystAgent(llm_api_key=self.api_key, rag_retriever=self.rag_retriever),
            MLEngineerAgent(llm_api_key=self.api_key, rag_retriever=self.rag_retriever),
            VisualizationAgent(llm_api_key=self.api_key, rag_retriever=self.rag_retriever),
            InsightAgent(llm_api_key=self.api_key, rag_retriever=self.rag_retriever)
        ]
        
        for agent in agents:
            self.orchestrator.register_agent(agent)
        
        self._initialized = True
        print("[MultiAgentPipeline] Ready with 5 agents")
    
    def run(
        self,
        file_path: str,
        target_column: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Run the multi-agent pipeline.
        
        Args:
            file_path: Path to the CSV/Excel file
            target_column: Name of the target column
            progress_callback: Optional Gradio progress callback
            
        Returns:
            Tuple of (HTML report, model path or None)
        """
        try:
            import time
            
            # Lazy initialize
            self._lazy_init()
            
            # Step 0: Load data
            if progress_callback is not None:
                progress_callback(0.05, desc="Loading data...")
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return "<h2>Error</h2><p>Unsupported file format. Use CSV or Excel.</p>", None
            
            if progress_callback is not None:
                progress_callback(0.1, desc="Data loaded successfully...")
                time.sleep(0.1)  # Small delay to show progress
            
            # Create initial state
            state = AgentState(
                data=df,
                target_column=target_column
            )
            
            # Step 1: Data Analyst
            if progress_callback is not None:
                progress_callback(0.15, desc="Cleaning data...")
                time.sleep(0.1)
            
            from agents import AgentRole
            if AgentRole.DATA_ANALYST in self.orchestrator.agents:
                state = self.orchestrator.agents[AgentRole.DATA_ANALYST].execute(state)
            
            if progress_callback is not None:
                progress_callback(0.25, desc="Data cleaning complete...")
                time.sleep(0.1)
            
            # Step 2: ML Engineer - Preparing
            if progress_callback is not None:
                progress_callback(0.3, desc="Preparing model training...")
                time.sleep(0.1)
            
            if progress_callback is not None:
                progress_callback(0.35, desc="Training model...")
                time.sleep(0.1)
            
            if AgentRole.ML_ENGINEER in self.orchestrator.agents:
                state = self.orchestrator.agents[AgentRole.ML_ENGINEER].execute(state)
            
            if progress_callback is not None:
                progress_callback(0.55, desc="Model training complete...")
                time.sleep(0.1)
            
            # Step 3: Visualization
            if progress_callback is not None:
                progress_callback(0.6, desc="Creating visualizations...")
                time.sleep(0.1)
            
            if AgentRole.VISUALIZATION in self.orchestrator.agents:
                state = self.orchestrator.agents[AgentRole.VISUALIZATION].execute(state)
            
            if progress_callback is not None:
                progress_callback(0.75, desc="Visualizations complete...")
                time.sleep(0.1)
            
            # Step 4: Insights
            if progress_callback is not None:
                progress_callback(0.8, desc="Generating AI insights...")
                time.sleep(0.1)
            
            if AgentRole.INSIGHT in self.orchestrator.agents:
                state = self.orchestrator.agents[AgentRole.INSIGHT].execute(state)
            
            if progress_callback is not None:
                progress_callback(0.9, desc="Insights generated...")
                time.sleep(0.1)
            
            # Generate report
            if progress_callback is not None:
                progress_callback(0.95, desc="Compiling final report...")
                time.sleep(0.1)
            
            report_html = self.orchestrator.generate_report(state)
            
            # Get model path if available
            model_path = getattr(state, 'model_path', None)
            
            if progress_callback is not None:
                progress_callback(1.0, desc="Complete!")
            
            return report_html, model_path
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[MultiAgentPipeline] Error: {tb}")
            error_html = f"""
            <h2>❌ Pipeline Error</h2>
            <p>{str(e)}</p>
            <details>
                <summary>Full Traceback</summary>
                <pre>{tb}</pre>
            </details>
            """
            return error_html, None


# Global instance for Gradio
_pipeline_instance = None


def get_pipeline() -> MultiAgentPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MultiAgentPipeline(initialize_rag=True)
    return _pipeline_instance


def run_multi_agent_pipeline(
    file_path: str,
    target_column: str,
    progress=None
) -> Tuple[str, Optional[str]]:
    """
    Convenience function to run the multi-agent pipeline.
    This is the function to call from Gradio.
    """
    pipeline = get_pipeline()
    return pipeline.run(file_path, target_column, progress)
