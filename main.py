import os
import io
import base64
import tempfile
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, 
                            mean_absolute_error, classification_report, 
                            confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import joblib
import time
import warnings
from datetime import datetime
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Load environment vars ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ======================================================
# 🧠 Groq LLM Call (GPT-OSS 120B)
# ======================================================
def call_gpt_oss_120b(prompt: str) -> str:
    """Call Groq GPT-OSS 120B for reasoning or explanations"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a data-science assistant that explains insights clearly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
        "reasoning_effort": "medium"
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM call failed: {e}"

# ======================================================
# 🧹 Data Cleaning and Preprocessing
# ======================================================
def clean_data(df: pd.DataFrame):
    """Enhanced data cleaning with better preprocessing"""
    df = df.copy()
    original_shape = df.shape
    
    # Drop columns that are all NaN
    df.dropna(axis=1, how="all", inplace=True)
    
    # Handle missing values more intelligently
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                # Use median for numeric columns
                df[col] = df[col].fillna(df[col].median())
            else:
                # Handle categorical columns
                mode_values = df[col].mode()
                if len(mode_values) > 0:
                    df[col] = df[col].fillna(mode_values[0])
    
    # Process categorical columns
    for col in df.select_dtypes(include="object").columns:
        # Try to detect date/time columns
        if df[col].dtype == 'object' and any(df[col].str.contains('-', na=False)):
            try:
                # Try to convert to datetime
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass
        
        # Encode categorical columns
        if df[col].dtype == 'object':
            # For high cardinality columns, use frequency encoding
            if df[col].nunique() > len(df[col]) / 2:
                freq_map = df[col].value_counts().to_dict()
                df[col] = df[col].map(freq_map)
            else:
                # For low cardinality, use label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    # Feature engineering for datetime columns
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            # Extract features from datetime
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_weekofyear'] = df[col].dt.isocalendar().week
            # Drop the original datetime column
            df.drop(col, axis=1, inplace=True)
    
    # Remove outliers for numeric columns (optional, can be toggled)
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df, f"Data cleaned: {original_shape} → {df.shape} rows/columns"

# ======================================================
# 📊 Enhanced Visualization
# ======================================================
def generate_visualizations(df: pd.DataFrame, target_col=None):
    """Generate comprehensive visualizations with proper encoding"""
    figs = []
    
    # 1. Data overview
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"Dataset Overview\n\nRows: {df.shape[0]}\nColumns: {df.shape[1]}\nTarget: {target_col}", 
             horizontalalignment='center', verticalalignment='center', fontsize=20)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    figs.append(("Dataset Overview", base64.b64encode(buf.read()).decode("utf-8")))
    plt.close()
    
    # 2. Target distribution
    if target_col and target_col in df.columns:
        plt.figure(figsize=(10, 6))
        if df[target_col].nunique() <= 20:  # Categorical
            sns.countplot(y=target_col, data=df)
            plt.title(f"Distribution of {target_col}")
        else:  # Continuous
            sns.histplot(df[target_col].dropna(), kde=True)
            plt.title(f"Distribution of {target_col}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        figs.append((f"Target Distribution ({target_col})", base64.b64encode(buf.read()).decode("utf-8")))
        plt.close()
    
    # 3. Correlation heatmap for numeric features
    numeric_cols = df.select_dtypes(include=np.number).columns[:10]  # Limit to first 10
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(12, 8))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        figs.append(("Feature Correlation", base64.b64encode(buf.read()).decode("utf-8")))
        plt.close()
    
    # 4. Feature distributions (top 5 numeric)
    for i, col in enumerate(df.select_dtypes(include=np.number).columns[:5]):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        figs.append((f"Feature {i+1}: {col}", base64.b64encode(buf.read()).decode("utf-8")))
        plt.close()
    
    return figs

# ======================================================
# 🤖 Enhanced Model Training
# ======================================================
def train_model(df, target_col, progress=None):
    """Enhanced model training with better algorithms and evaluation"""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Normalize column names (strip + lower)
    df.columns = df.columns.str.strip().str.lower()
    target_col = target_col.strip().lower()

    # Find the target column
    if target_col not in df.columns:
        similar = [col for col in df.columns if target_col in col]
        if similar:
            target_col = similar[0]
        else:
            raise ValueError(
                f"Target column '{target_col}' not found.\nAvailable columns: {list(df.columns)}"
            )

    # Split features/target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Select only numeric columns for features
    X = X.select_dtypes(include=["number"]).fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No numeric columns found for training.")

    # Determine if this is a classification or regression task
    is_classification = False
    if y.dtype == 'object' or y.nunique() < 20:  # Heuristic for classification
        is_classification = True
        # Encode target if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
    
    # Check for class imbalance
    is_imbalanced = False
    if is_classification:
        class_counts = np.bincount(y)
        min_class = np.min(class_counts)
        max_class = np.max(class_counts)
        is_imbalanced = min_class / max_class < 0.2  # If minority class is less than 20% of majority
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if is_classification else None)
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance if needed
    if is_imbalanced and is_classification:
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Apply SMOTE for oversampling
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        except:
            # Fallback if SMOTE fails
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
        class_weight_dict = None
    
    # Model selection and hyperparameter tuning
    if progress:
        progress(0.5, desc="Training and tuning models...")
    
    if is_classification:
        # Try multiple classification models
        models = {
            'RandomForest': RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, class_weight=class_weight_dict, max_iter=1000)
        }
        
        # Hyperparameter grids
        param_grids = {
            'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
            'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'LogisticRegression': {'C': [0.1, 1.0, 10.0]}
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            # Simple grid search with cross-validation
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_weighted')
            grid_search.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            if f1 > best_score:
                best_score = f1
                best_model = grid_search.best_estimator_
                best_name = name
        
        # Final evaluation
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Convert numpy types to native Python types for JSON serialization
        metric = {
            "Model": best_name,
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1 Score": float(f1),
            "Imbalanced": bool(is_imbalanced),
            "Classes": int(len(np.unique(y)))
        }
    else:
        # Try multiple regression models
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        # Hyperparameter grids
        param_grids = {
            'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
            'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'LinearRegression': {}  # No hyperparameters for LinearRegression
        }
        
        best_model = None
        best_score = float('inf')
        best_name = ""
        
        for name, model in models.items():
            # Simple grid search with cross-validation
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            
            if mse < best_score:
                best_score = mse
                best_model = grid_search.best_estimator_
                best_name = name
        
        # Final evaluation
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Convert numpy types to native Python types for JSON serialization
        metric = {
            "Model": best_name,
            "MSE": float(mse),
            "MAE": float(mae),
            "R2": float(r2)
        }

    # Save the model and scaler
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{target_col}_model.pkl"
    scaler_path = f"models/{target_col}_scaler.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        importance_path = f"models/{target_col}_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        
        return model_path, metric, feature_importance.head(10).to_html(index=False)
    
    return model_path, metric, None

# ======================================================
# 🧾 Enhanced Full Pipeline
# ======================================================
def run_pipeline(file_obj, target_col, progress=gr.Progress()):
    try:
        # Check if file is None
        if file_obj is None:
            return "<h2>Error</h2><p>Please upload a file.</p>", None
        
        # Read the file - handle both file path and file object
        if isinstance(file_obj, str):
            # If it's a file path
            if file_obj.endswith('.csv'):
                df = pd.read_csv(file_obj)
            else:
                df = pd.read_excel(file_obj)
        else:
            # If it's a file object from Gradio
            file_extension = os.path.splitext(file_obj.name)[1].lower() if hasattr(file_obj, 'name') else ''
            
            if file_extension == '.csv':
                df = pd.read_csv(file_obj)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_obj)
            else:
                # Try to read as CSV by default
                df = pd.read_csv(file_obj)
        
        # Show data preview
        preview = df.head().to_html(classes="table table-striped")
        
        progress(0.1, desc="Cleaning data...")
        
        # Clean the data
        cleaned, cleaning_msg = clean_data(df)
        
        progress(0.3, desc="Generating visualizations...")
        
        # Generate visualizations
        figs = generate_visualizations(cleaned, target_col)
        
        progress(0.5, desc="Training model...")
        
        # Train the model
        model_path, metric, importance_html = train_model(cleaned, target_col, progress)
        
        progress(0.8, desc="Generating insights...")
        
        # Generate insights using LLM
        insight_prompt = f"""
        The dataset has {cleaned.shape[0]} rows and {cleaned.shape[1]} columns.
        Target column: {target_col}.
        Model performance: {metric}.
        Key insight summary in 5 sentences.
        """
        llm_summary = call_gpt_oss_120b(insight_prompt)

        # Create HTML report with properly formatted metrics
        metrics_html = "<ul>"
        for key, value in metric.items():
            if isinstance(value, float):
                metrics_html += f"<li><strong>{key}:</strong> {value:.4f}</li>"
            else:
                metrics_html += f"<li><strong>{key}:</strong> {value}</li>"
        metrics_html += "</ul>"
        
        html = f"""
        <h2>🧠 Data Pipeline Report</h2>
        <h3>Data Cleaning</h3>
        <p>{cleaning_msg}</p>
        
        <h3>Data Preview</h3>
        {preview}
        
        <h3>Model Performance</h3>
        {metrics_html}
        """
        
        if importance_html:
            html += f"""
            <h3>Feature Importance</h3>
            {importance_html}
            """
        
        html += f"""
        <h3>AI Insights</h3>
        <p>{llm_summary}</p>
        
        <h3>Visualizations</h3>
        """
        
        # Add visualizations
        for title, img in figs:
            html += f"<h4>{title}</h4><img src='data:image/png;base64,{img}' width='600'/>"

        return html, model_path
    except Exception as e:
        error_html = f"<h2>Error in Pipeline</h2><p>{str(e)}</p>"
        return error_html, None

# ======================================================
# 🎨 Gradio App Creation
# ======================================================
def create_gradio_app():
    """Create and return the Gradio app"""
    # Create a custom CSS to hide PWA elements
    custom_css = """
    /* Hide PWA install button */
    .install-pwa-btn {
        display: none !important;
    }
    
    /* Hide update banner */
    .update-banner {
        display: none !important;
    }
    
    /* Fix container width */
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="Data Pipeline Agent",
        analytics_enabled=False,  # Disable analytics
        css=custom_css
    ) as app:
        gr.Markdown("# 🧠 Data Pipeline Agent (Groq GPT-OSS 120B)")
        gr.Markdown("Upload dataset → Auto-clean → Visualize → Train → Generate Report")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="📁 Upload Dataset (CSV/Excel)")
                target_input = gr.Textbox(label="🎯 Target Column Name", placeholder="e.g., price, target, label")
                submit_btn = gr.Button("Run Pipeline", variant="primary")
            
            with gr.Column(scale=2):
                html_output = gr.HTML(label="📊 Report")
                model_output = gr.File(label="💾 Download Trained Model (.pkl)")
        
        # Event handler
        submit_btn.click(
            fn=run_pipeline,
            inputs=[file_input, target_input],
            outputs=[html_output, model_output]
        )

            # Set max_file_size attribute to prevent error
    app.max_file_size = 50 * 1024 * 1024  # 50MB

 # Add queue support
    app.queue()
    return app

# Function to return Gradio app for FastAPI mounting
def gradio_ui():
    """Return the Gradio app for mounting in FastAPI"""
    return create_gradio_app()

# Keep this for standalone execution
# Keep this for standalone execution
# Keep this for standalone execution
# Keep this for standalone execution
# Keep this for standalone execution
if __name__ == "__main__":
    import os
    from fastapi import FastAPI, Response
    import uvicorn
    
    # Create the Gradio app
    app = create_gradio_app()
    
    # Add custom routes directly to Gradio app
    @app.get("/manifest.json")
    async def get_manifest():
        """Serve PWA manifest"""
        return Response(content='{"name": "Data Pipeline Agent"}', media_type="application/json")
    
    @app.get("/gradio_api/upload_progress")
    async def upload_progress():
        return {"status": "completed", "progress": 100}
    
    # Launch the app
    app.queue(concurrency_count=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,
        show_error=True,
        quiet=False,
        inbrowser=False
    )