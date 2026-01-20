"""
Orchestrator Agent - Coordinates all agents in the multi-agent system.
Routes tasks, manages state, and generates final reports.
"""
from typing import Optional, Any, Callable
from .base_agent import BaseAgent, AgentRole, AgentState


class OrchestratorAgent(BaseAgent):
    """
    The Orchestrator Agent coordinates the entire data pipeline.
    
    Responsibilities:
    - Route tasks to specialized agents
    - Manage shared state between agents
    - Handle errors and fallbacks
    - Generate final consolidated report
    """
    
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        rag_retriever: Optional[Any] = None
    ):
        super().__init__(
            role=AgentRole.ORCHESTRATOR,
            llm_api_key=llm_api_key,
            rag_retriever=rag_retriever
        )
        self.agents = {}
        self.progress_callback: Optional[Callable] = None
    
    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent with the orchestrator."""
        self.agents[agent.role] = agent
        self.log(f"Registered agent: {agent.role.value}")
    
    def set_progress_callback(self, callback: Callable):
        """Set a callback for progress updates (for Gradio UI)."""
        self.progress_callback = callback
    
    def update_progress(self, progress: float, message: str):
        """Update progress if callback is set."""
        if self.progress_callback is not None:
            try:
                self.progress_callback(progress, desc=message)
            except Exception as e:
                print(f"[ORCHESTRATOR] Progress update failed: {e}")
        self.log(f"[{int(progress*100)}%] {message}")
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the full multi-agent pipeline.
        
        Flow:
        1. Data Analyst → Clean and analyze data
        2. ML Engineer → Train and evaluate models
        3. Visualization → Generate charts
        4. Insight → Generate AI-powered insights
        5. Compile final report
        """
        try:
            state.current_step = "orchestrating"
            self.update_progress(0.1, "Starting multi-agent pipeline...")
            
            # Step 1: Data Analysis
            if AgentRole.DATA_ANALYST in self.agents:
                self.update_progress(0.2, "Data Analyst analyzing data...")
                state = self.agents[AgentRole.DATA_ANALYST].execute(state)
            
            # Step 2: ML Engineering
            if AgentRole.ML_ENGINEER in self.agents:
                self.update_progress(0.4, "ML Engineer training models...")
                state = self.agents[AgentRole.ML_ENGINEER].execute(state)
            
            # Step 3: Visualization
            if AgentRole.VISUALIZATION in self.agents:
                self.update_progress(0.6, "Visualization Agent creating charts...")
                state = self.agents[AgentRole.VISUALIZATION].execute(state)
            
            # Step 4: Insights
            if AgentRole.INSIGHT in self.agents:
                self.update_progress(0.8, "Insight Agent generating insights...")
                state = self.agents[AgentRole.INSIGHT].execute(state)
            
            # Final step: Compile report
            self.update_progress(0.95, "Compiling final report...")
            state.current_step = "completed"
            
            return state
            
        except Exception as e:
            state.errors.append(f"Orchestrator error: {str(e)}")
            state.current_step = "error"
            self.log(f"Pipeline failed: {e}")
            return state
    
    def generate_report(self, state: AgentState) -> str:
        """Generate the final HTML report from agent results."""
        html = """
        <h2>🧠 Multi-Agent Data Pipeline Report</h2>
        <p><em>Powered by AI Agents with RAG</em></p>
        """
        
        # Data Cleaning Section
        if state.cleaning_report:
            html += f"""
            <h3>📊 Data Analysis</h3>
            <p>{state.cleaning_report}</p>
            """
        
        # Model Results Section
        if state.model_results:
            html += "<h3>🤖 Model Performance</h3><ul>"
            for key, value in state.model_results.items():
                if isinstance(value, float):
                    html += f"<li><strong>{key}:</strong> {value:.4f}</li>"
                else:
                    html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        # Insights Section
        if state.insights:
            html += f"""
            <h3>💡 AI Insights</h3>
            <p>{state.insights}</p>
            """
        
        # Visualizations Section
        if state.visualizations:
            html += "<h3>📈 Visualizations</h3>"
            for title, img_data in state.visualizations:
                html += f"<h4>{title}</h4>"
                html += f"<img src='data:image/png;base64,{img_data}' width='600'/>"
        
        # Errors Section (if any)
        if state.errors:
            html += "<h3>⚠️ Warnings</h3><ul>"
            for error in state.errors:
                html += f"<li>{error}</li>"
            html += "</ul>"
        
        return html
