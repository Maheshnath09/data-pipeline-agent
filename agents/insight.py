"""
Insight Agent - Generates AI-powered insights using RAG for context.
"""
from typing import Optional, Any
from .base_agent import BaseAgent, AgentRole, AgentState


class InsightAgent(BaseAgent):
    """
    Insight Agent - Generates human-readable, actionable insights.
    
    Capabilities:
    - Summarize analysis results
    - Explain model decisions
    - Provide recommendations
    - Use RAG for domain-specific context
    """
    
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        rag_retriever: Optional[Any] = None
    ):
        super().__init__(
            role=AgentRole.INSIGHT,
            llm_api_key=llm_api_key,
            rag_retriever=rag_retriever
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """Generate insights from the analysis and model results."""
        self.log("Generating insights...")
        
        try:
            # Build context from RAG
            rag_context = ""
            if self.rag_retriever:
                # Query for relevant best practices
                model_type = state.model_results.get('Type', 'classification')
                rag_context = self.rag_retriever.get_context(
                    f"Best practices for {model_type} model interpretation and recommendations",
                    k=3
                )
            
            # Build the insight prompt
            prompt = self._build_insight_prompt(state, rag_context)
            
            # Generate insights using LLM
            insights = self.call_llm(prompt, max_tokens=800)
            
            # Update state
            state.insights = insights
            state.current_step = "insights_generated"
            
            self.log("Insights generated successfully")
            
        except Exception as e:
            state.errors.append(f"Insight generation error: {str(e)}")
            state.insights = "Unable to generate insights. Please review the analysis results directly."
            self.log(f"Error: {e}")
        
        return state
    
    def _build_insight_prompt(self, state: AgentState, rag_context: str) -> str:
        """Build the prompt for insight generation."""
        # Extract key information
        model_info = state.model_results or {}
        cleaning_info = state.cleaning_report or "No cleaning information available"
        
        # Format model metrics
        metrics_str = ""
        for key, value in model_info.items():
            if isinstance(value, float):
                metrics_str += f"- {key}: {value:.4f}\n"
            else:
                metrics_str += f"- {key}: {value}\n"
        
        prompt = f"""You are an expert data scientist providing insights on a machine learning analysis.

## Analysis Summary

### Data Cleaning
{cleaning_info}

### Model Performance
{metrics_str}

### Reference Context (from knowledge base)
{rag_context if rag_context else "No additional context available."}

## Your Task

Generate a comprehensive insight report with the following sections:

1. **Key Findings** (2-3 bullet points)
   - What are the most important discoveries from this analysis?

2. **Model Performance Assessment** (2-3 sentences)
   - How well did the model perform?
   - Are there any concerns about the metrics?

3. **Recommendations** (3-4 bullet points)
   - What actions should be taken based on these results?
   - How could the analysis be improved?

4. **Next Steps** (2-3 bullet points)
   - What should be done to deploy or improve this model?

Be concise, specific, and actionable. Use markdown formatting."""

        return prompt
