# Data Analysis Guidelines

## Data Quality Assessment

### Initial Data Checks
1. Check dimensions (rows, columns)
2. Identify data types (numeric, categorical, datetime, text)
3. Calculate missing value percentages per column
4. Detect duplicate rows
5. Check for constant columns (no variance)

### Common Data Issues

#### Missing Data Patterns
- **MCAR (Missing Completely at Random)**: Safe to impute or drop
- **MAR (Missing at Random)**: Impute based on other features
- **MNAR (Missing Not at Random)**: Requires domain knowledge

#### Data Type Issues
- Numeric stored as string (often due to currency symbols, commas)
- Dates in multiple formats within same column
- Mixed types in categorical columns

### Column Analysis by Type

#### Numeric Columns
- Calculate: mean, median, std, min, max, quartiles
- Check for: outliers, skewness, zero-inflation
- Visualize: histogram, box plot

#### Categorical Columns
- Count unique values
- Check cardinality (high-cardinality needs special handling)
- Identify rare categories
- Visualize: bar chart, value counts

#### Datetime Columns
- Check range (start, end dates)
- Identify gaps or missing periods
- Check timezone consistency
- Extract useful components (year, month, hour, weekday)

#### Text Columns
- Calculate average length
- Check for empty strings vs NaN
- Identify potential categories (low unique count)
- Consider: language detection, sentiment if needed

## Target Variable Analysis

### Classification Targets
- Calculate class distribution
- Identify imbalance: if minority <10%, consider balancing
- Check for rare classes that may need grouping

### Regression Targets
- Check distribution (normal, skewed, multimodal)
- Identify outliers in target
- Consider log transformation for skewed targets
- Check for zero-inflation

## Feature-Target Relationships

### Numeric Features vs Classification Target
- Compare distributions across classes (box plot)
- Calculate correlation ratio or ANOVA F-score
- Check for feature importance ranking

### Numeric Features vs Regression Target
- Calculate Pearson/Spearman correlation
- Visualize with scatter plot
- Check for non-linear relationships

### Categorical Features vs Target
- Chi-square test for classification
- ANOVA for regression
- Visualize with grouped bar charts or heatmaps
