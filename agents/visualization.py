"""
Visualization Agent - Creates intelligent, context-aware visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional, Any, List, Tuple
from .base_agent import BaseAgent, AgentRole, AgentState


class VisualizationAgent(BaseAgent):
    """
    Visualization Agent - Creates meaningful data visualizations.
    
    Capabilities:
    - Automatic chart type selection
    - Target distribution visualization
    - Correlation analysis
    - Feature importance plots
    """
    
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        rag_retriever: Optional[Any] = None
    ):
        super().__init__(
            role=AgentRole.VISUALIZATION,
            llm_api_key=llm_api_key,
            rag_retriever=rag_retriever
        )
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def execute(self, state: AgentState) -> AgentState:
        """Generate visualizations based on data and model results."""
        self.log("Generating visualizations...")
        
        try:
            df = state.data
            target_col = state.target_column.strip().lower()
            
            if df is None:
                state.errors.append("No data available for visualization")
                return state
            
            visualizations = []
            
            # 1. Dataset Overview
            viz = self._create_overview(df, target_col)
            if viz:
                visualizations.append(viz)
            
            # 2. Target Distribution
            if target_col in df.columns:
                viz = self._create_target_distribution(df, target_col)
                if viz:
                    visualizations.append(viz)
            
            # 3. Correlation Heatmap
            viz = self._create_correlation_heatmap(df)
            if viz:
                visualizations.append(viz)
            
            # 4. Feature Distributions (top 4)
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
            for col in numeric_cols:
                viz = self._create_feature_distribution(df, col)
                if viz:
                    visualizations.append(viz)
            
            # Update state
            state.visualizations = visualizations
            state.current_step = "visualizations_created"
            
            self.log(f"Created {len(visualizations)} visualizations")
            
        except Exception as e:
            state.errors.append(f"Visualization error: {str(e)}")
            self.log(f"Error: {e}")
        
        return state
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    def _create_overview(self, df: pd.DataFrame, target_col: str) -> Tuple[str, str]:
        """Create dataset overview visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create summary text
        summary_text = f"""
        📊 Dataset Overview
        
        Rows: {df.shape[0]:,}
        Columns: {df.shape[1]}
        Target: {target_col}
        
        Missing Values: {df.isnull().sum().sum():,}
        Numeric Features: {len(df.select_dtypes(include=[np.number]).columns)}
        Categorical Features: {len(df.select_dtypes(include=['object', 'category']).columns)}
        """
        
        ax.text(0.5, 0.5, summary_text, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14,
                transform=ax.transAxes,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax.axis('off')
        ax.set_title('Dataset Summary', fontsize=16, fontweight='bold')
        
        return ("Dataset Overview", self._fig_to_base64(fig))
    
    def _create_target_distribution(self, df: pd.DataFrame, target_col: str) -> Tuple[str, str]:
        """Create target variable distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if df[target_col].nunique() <= 20:  # Categorical
                value_counts = df[target_col].value_counts().head(15)
                sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')
                ax.set_xlabel('Count')
                ax.set_ylabel(target_col)
            else:  # Continuous
                sns.histplot(df[target_col].dropna(), kde=True, ax=ax, color='steelblue')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Frequency')
            
            ax.set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
            
            return (f"Target: {target_col}", self._fig_to_base64(fig))
            
        except Exception as e:
            plt.close(fig)
            self.log(f"Target distribution error: {e}")
            return None
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """Create correlation heatmap for numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
        
        # Limit to top 12 columns
        if numeric_df.shape[1] > 12:
            numeric_df = numeric_df.iloc[:, :12]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr, 
            mask=mask,
            annot=True, 
            fmt=".2f",
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        return ("Correlation Heatmap", self._fig_to_base64(fig))
    
    def _create_feature_distribution(self, df: pd.DataFrame, col: str) -> Tuple[str, str]:
        """Create distribution plot for a numeric feature."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        try:
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='coral')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            
            # Add statistics annotation
            stats_text = f'Mean: {df[col].mean():.2f}\nStd: {df[col].std():.2f}'
            ax.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                       fontsize=10, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            return (f"Feature: {col}", self._fig_to_base64(fig))
            
        except Exception as e:
            plt.close(fig)
            self.log(f"Feature distribution error for {col}: {e}")
            return None
