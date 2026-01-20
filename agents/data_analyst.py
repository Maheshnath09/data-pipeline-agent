"""
Data Analyst Agent - Analyzes and cleans data with RAG-powered insights.
"""
import pandas as pd
import numpy as np
from typing import Optional, Any
from .base_agent import BaseAgent, AgentRole, AgentState


class DataAnalystAgent(BaseAgent):
    """
    Data Analyst Agent - Responsible for data understanding and cleaning.
    
    Capabilities:
    - Detect data types and patterns
    - Clean and preprocess data
    - Handle missing values intelligently
    - Feature engineering suggestions
    """
    
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        rag_retriever: Optional[Any] = None
    ):
        super().__init__(
            role=AgentRole.DATA_ANALYST,
            llm_api_key=llm_api_key,
            rag_retriever=rag_retriever
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """Analyze and clean the data."""
        self.log("Starting data analysis...")
        
        try:
            df = state.data
            if df is None or not isinstance(df, pd.DataFrame):
                state.errors.append("No valid DataFrame provided")
                return state
            
            # Analyze data
            analysis = self._analyze_data(df, state.target_column)
            
            # Clean data based on analysis
            cleaned_df, cleaning_actions = self._clean_data(df, state.target_column)
            
            # Get RAG context for recommendations
            if self.rag_retriever:
                context = self.rag_retriever.get_context(
                    f"How to clean data with {analysis['missing_pct']:.1f}% missing values and {analysis['num_categorical']} categorical columns",
                    k=2
                )
                # Use context for advice (logged for now)
                self.log(f"RAG context: {context[:100]}...")
            
            # Generate summary using LLM
            summary_prompt = f"""Summarize this data analysis in 2-3 sentences:
- Dataset: {df.shape[0]} rows, {df.shape[1]} columns
- Target: {state.target_column}
- Missing: {analysis['missing_pct']:.1f}%
- Numeric columns: {analysis['num_numeric']}
- Categorical columns: {analysis['num_categorical']}
- Actions taken: {', '.join(cleaning_actions)}"""
            
            cleaning_summary = self.call_llm(summary_prompt, max_tokens=200)
            
            # Update state
            state.data = cleaned_df
            state.cleaning_report = cleaning_summary
            state.current_step = "data_analyzed"
            
            self.log(f"Analysis complete. Cleaned shape: {cleaned_df.shape}")
            
        except Exception as e:
            state.errors.append(f"Data analysis error: {str(e)}")
            self.log(f"Error: {e}")
        
        return state
    
    def _analyze_data(self, df: pd.DataFrame, target_col: str) -> dict:
        """Analyze data characteristics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        
        return {
            'num_numeric': len(numeric_cols),
            'num_categorical': len(categorical_cols),
            'missing_pct': missing_pct,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'target_type': 'categorical' if df[target_col].dtype == 'object' or df[target_col].nunique() < 20 else 'numeric'
        }
    
    def _clean_data(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Clean and preprocess the data."""
        df = df.copy()
        actions = []
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        target_col = target_col.strip().lower()
        actions.append("normalized column names")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            actions.append(f"removed {initial_rows - len(df)} duplicates")
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    df[col] = df[col].fillna(df[col].median())
                    actions.append(f"imputed {col} with median")
                else:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                        actions.append(f"imputed {col} with mode")
        
        # Handle datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df = df.drop(columns=[col])
                    actions.append(f"extracted datetime features from {col}")
                except:
                    pass
        
        return df, actions
