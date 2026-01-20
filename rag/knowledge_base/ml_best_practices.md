# Machine Learning Best Practices

## Data Cleaning

### Handling Missing Values
- For numeric columns with <5% missing: use median imputation (robust to outliers)
- For numeric columns with >5% missing: consider creating a "missing" indicator feature
- For categorical columns: use mode imputation or create "Unknown" category
- Never impute with mean for skewed distributions

### Outlier Detection
- Use IQR method for univariate outliers
- For multivariate outliers, consider Isolation Forest or DBSCAN
- Always investigate outliers before removing - they might be valid edge cases

### Feature Scaling
- StandardScaler: when features are normally distributed
- MinMaxScaler: when you need values in [0,1] range
- RobustScaler: when data has many outliers

## Model Selection

### Classification Tasks
- **Small dataset (<1000 samples)**: Logistic Regression or SVM
- **Medium dataset (1000-100k samples)**: Random Forest or Gradient Boosting
- **Large dataset (>100k samples)**: XGBoost, LightGBM, or Neural Networks
- **Imbalanced classes**: Use SMOTE, class weights, or ensemble methods

### Regression Tasks
- **Linear relationships**: Linear/Ridge/Lasso Regression
- **Non-linear relationships**: Random Forest, XGBoost, or Neural Networks
- **Many features**: Use regularization (L1/L2) or feature selection

### Handling Class Imbalance
- SMOTE oversampling works well when minority class has >100 samples
- Class weights are more memory efficient than oversampling
- For extreme imbalance (>1:100), consider anomaly detection approach

## Feature Engineering

### Datetime Features
- Extract: year, month, day, hour, day_of_week, is_weekend
- Create cyclical features using sin/cos for periodic patterns
- Calculate time differences between related events

### Text Features
- TF-IDF for traditional ML models
- Word embeddings (Word2Vec, FastText) for semantic similarity
- Sentence transformers for document-level representations

### Categorical Features
- Label encoding: for ordinal categories
- One-hot encoding: for nominal categories with <10 unique values
- Target encoding: for high-cardinality categories (with proper cross-validation)

## Model Evaluation

### Classification Metrics
- **Accuracy**: Only when classes are balanced
- **Precision**: When false positives are costly
- **Recall**: When false negatives are costly
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall model discrimination ability

### Regression Metrics
- **MSE/RMSE**: Penalizes large errors heavily
- **MAE**: More robust to outliers
- **R²**: Proportion of variance explained
- **MAPE**: Easy to interpret as percentage error
