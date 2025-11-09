# 🧠 Data Pipeline Agent

An AI-powered data processing pipeline that automatically cleans, visualizes, and trains machine learning models on your datasets with intelligent insights powered by Groq's GPT-OSS 120B model.

## ✨ Features

- **🧹 Intelligent Data Cleaning**: Automatic handling of missing values, outliers, and categorical encoding
- **📊 Comprehensive Visualizations**: Auto-generated charts including correlation matrices, distributions, and target analysis
- **🤖 Smart Model Training**: Automatic model selection between classification/regression with hyperparameter tuning
- **🧠 AI-Powered Insights**: LLM-generated explanations and recommendations using Groq GPT-OSS 120B
- **⚖️ Class Imbalance Handling**: SMOTE oversampling and class weighting for imbalanced datasets
- **📱 Progressive Web App**: PWA support with offline capabilities
- **🚀 FastAPI Backend**: RESTful API with async processing and background tasks
- **🎨 Modern Web UI**: Clean, responsive interface with real-time progress tracking

## 🏗️ Architecture

```
data_pipeline_agent/
├── api.py              # FastAPI backend with REST endpoints
├── main.py             # Core ML pipeline and Gradio interface
├── models/             # Saved trained models and artifacts
├── static/             # Static assets (favicon, icons)
├── templates/          # HTML templates (if any)
├── Dockerfile          # Container configuration
├── pyproject.toml      # Python dependencies
├── manifest.json       # PWA manifest
├── sw.js              # Service worker for PWA
└── .env               # Environment variables
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Groq API key (for AI insights)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data_pipeline_agent
   ```

2. **Install dependencies using uv (recommended)**
   ```bash
   pip install uv
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   # FastAPI backend (recommended for production)
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   
   # Or Gradio standalone
   python main.py
   ```

5. **Access the application**
   - FastAPI + Custom UI: http://localhost:8000
   - Gradio Interface: http://localhost:8000/gradio
   - API Documentation: http://localhost:8000/docs

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t data-pipeline-agent .

# Run the container
docker run -p 8000:8000 --env-file .env data-pipeline-agent
```

## 📡 API Endpoints

### Core Pipeline Endpoints

- `POST /upload` - Upload dataset file
- `POST /run_pipeline/{file_id}` - Start pipeline processing
- `GET /status/{file_id}` - Check processing status
- `GET /report/{file_id}` - Get HTML report
- `GET /download/{file_id}` - Download trained model

### Static & PWA Endpoints

- `GET /static/*` - Static file serving
- `GET /favicon.ico` - Favicon
- `GET /manifest.json` - PWA manifest
- `GET /sw.js` - Service worker

## 🔧 Usage

### 1. Web Interface

1. Navigate to http://localhost:8000
2. Upload your CSV/Excel dataset
3. Specify the target column name
4. Click "Run Pipeline"
5. Monitor progress and view results
6. Download the trained model

### 2. API Usage

```python
import requests

# Upload file
with open('dataset.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
    file_id = response.json()['file_id']

# Run pipeline
response = requests.post(f'http://localhost:8000/run_pipeline/{file_id}', 
                        params={'target_column': 'price'})

# Check status
status = requests.get(f'http://localhost:8000/status/{file_id}').json()

# Get report when completed
if status['status'] == 'completed':
    report = requests.get(f'http://localhost:8000/report/{file_id}').text
```

## 🧠 AI Features

### Intelligent Data Processing

- **Missing Value Handling**: Median for numeric, mode for categorical
- **Categorical Encoding**: Frequency encoding for high cardinality, label encoding for low
- **DateTime Feature Engineering**: Automatic extraction of time-based features
- **Outlier Detection**: IQR-based clipping for numeric columns

### Smart Model Selection

**Classification Tasks:**
- Random Forest Classifier
- Gradient Boosting Classifier  
- Logistic Regression

**Regression Tasks:**
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression

### Advanced Features

- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Class Imbalance**: SMOTE oversampling + class weighting
- **Feature Importance**: Automatic extraction and ranking
- **Model Persistence**: Joblib serialization with scalers

## 📊 Supported Data Formats

- **CSV files** (`.csv`)
- **Excel files** (`.xlsx`, `.xls`)
- **Automatic encoding detection**
- **Mixed data types support**

## 🔍 Generated Insights

The pipeline automatically generates:

1. **Data Overview**: Shape, types, missing values
2. **Target Distribution**: Histograms and count plots
3. **Correlation Analysis**: Feature correlation heatmaps
4. **Feature Distributions**: Individual feature analysis
5. **Model Performance**: Comprehensive metrics
6. **Feature Importance**: Top contributing features
7. **AI Explanations**: LLM-powered insights and recommendations

## ⚙️ Configuration

### Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key_here  # Required for AI insights
```

### Model Parameters

Models are automatically tuned with these parameter grids:

```python
# Random Forest
{'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}

# Gradient Boosting  
{'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}

# Logistic Regression
{'C': [0.1, 1.0, 10.0]}
```

## 📈 Performance Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Class imbalance detection

### Regression  
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

## 🛠️ Development

### Project Structure

```python
# Core modules
main.py          # ML pipeline logic
api.py           # FastAPI backend
models/          # Saved models directory

# Dependencies
pyproject.toml   # Python dependencies (uv format)
requirements.txt # Pip format (if needed)

# Deployment
Dockerfile       # Multi-stage container build
.env            # Environment configuration
```

### Adding New Models

```python
# In train_model() function
models = {
    'YourModel': YourModelClass(params),
    # Add hyperparameter grid
}

param_grids = {
    'YourModel': {'param1': [val1, val2], 'param2': [val3, val4]}
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Groq API Key Missing**
   ```
   Error: LLM call failed
   Solution: Set GROQ_API_KEY in .env file
   ```

2. **Target Column Not Found**
   ```
   Error: Target column 'xyz' not found
   Solution: Check column names, case-sensitive matching
   ```

3. **No Numeric Features**
   ```
   Error: No numeric columns found for training
   Solution: Ensure dataset has numeric features
   ```

### Performance Tips

- **Large Datasets**: Consider sampling for faster processing
- **High Cardinality**: Categorical features are automatically handled
- **Memory Usage**: Models and scalers are saved to disk
- **Concurrent Processing**: FastAPI handles multiple requests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the troubleshooting section above

---

**Built with ❤️ using FastAPI, Scikit-learn, and Groq AI**