"""
ML Engineer Agent - Handles model selection, training, and evaluation.
"""
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict
from .base_agent import BaseAgent, AgentRole, AgentState

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


class MLEngineerAgent(BaseAgent):
    """
    ML Engineer Agent - Responsible for model training and optimization.
    
    Capabilities:
    - Automatic model selection based on data characteristics
    - Hyperparameter tuning
    - Model evaluation and comparison
    - Feature importance analysis
    """
    
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        rag_retriever: Optional[Any] = None
    ):
        super().__init__(
            role=AgentRole.ML_ENGINEER,
            llm_api_key=llm_api_key,
            rag_retriever=rag_retriever
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """Train and evaluate ML models."""
        self.log("Starting model training...")
        
        try:
            df = state.data
            target_col = state.target_column.strip().lower()
            
            if df is None or target_col not in df.columns:
                # Try fuzzy match
                matches = [c for c in df.columns if target_col in c]
                if matches:
                    target_col = matches[0]
                else:
                    state.errors.append(f"Target column '{target_col}' not found")
                    return state
            
            # Prepare data
            X, y, is_classification = self._prepare_data(df, target_col)
            
            if X.shape[1] == 0:
                state.errors.append("No numeric features found for training")
                return state
            
            # Get RAG advice on model selection
            if self.rag_retriever:
                task_type = "classification" if is_classification else "regression"
                context = self.rag_retriever.get_context(
                    f"Best {task_type} model for {len(df)} samples with {X.shape[1]} features",
                    k=2
                )
                self.log(f"RAG advice: {context[:100]}...")
            
            # Train models
            results, best_model, model_name = self._train_models(X, y, is_classification)
            
            # Save model
            model_path = f"models/{target_col}_model.pkl"
            self._save_model(best_model, model_path)
            
            # Get feature importance
            importance_html = self._get_feature_importance(best_model, X.columns.tolist())
            
            # Update state
            state.model_results = {
                'Model': model_name,
                'Type': 'Classification' if is_classification else 'Regression',
                **results
            }
            state.model_path = model_path
            state.importance_html = importance_html
            state.current_step = "model_trained"
            
            self.log(f"Training complete. Best model: {model_name}")
            
        except Exception as e:
            state.errors.append(f"ML training error: {str(e)}")
            self.log(f"Error: {e}")
        
        return state
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> tuple:
        """Prepare features and target for training."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Keep only numeric features
        X = X.select_dtypes(include=[np.number]).fillna(0)
        
        # Determine task type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        
        # Encode target if classification
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        return X, y, is_classification
    
    def _train_models(self, X, y, is_classification: bool) -> tuple:
        """Train multiple models and select the best one."""
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if is_classification else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        if is_classification:
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
            }
            scoring = 'accuracy'
        else:
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Ridge': Ridge(random_state=42)
            }
            scoring = 'r2'
        
        # Train and evaluate
        best_score = -np.inf
        best_model = None
        best_name = None
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring=scoring)
                mean_score = cv_scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    model.fit(X_train_scaled, y_train)
                    best_model = model
                    best_name = name
                    
                self.log(f"{name}: CV score = {mean_score:.4f}")
                
            except Exception as e:
                self.log(f"{name} failed: {e}")
        
        # Get final metrics
        y_pred = best_model.predict(X_test_scaled)
        
        if is_classification:
            results = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1_Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            results = {
                'R2_Score': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
        
        return results, best_model, best_name
    
    def _save_model(self, model, path: str):
        """Save the trained model."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        self.log(f"Model saved to {path}")
    
    def _get_feature_importance(self, model, feature_names: list) -> str:
        """Generate feature importance HTML."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:10]  # Top 10
            
            html = "<ul>"
            for idx in sorted_idx:
                if idx < len(feature_names):
                    html += f"<li><strong>{feature_names[idx]}:</strong> {importances[idx]:.4f}</li>"
            html += "</ul>"
            return html
        
        return ""
