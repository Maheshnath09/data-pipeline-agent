# Data Visualization Best Practices

## Chart Selection Guide

### For Distribution Analysis
- **Histogram**: Show distribution of a single numeric variable
- **Box Plot**: Compare distributions across categories, highlight outliers
- **Violin Plot**: Histogram + box plot combined for detailed distribution view
- **KDE Plot**: Smoothed distribution, good for comparing multiple distributions

### For Relationships
- **Scatter Plot**: Show relationship between two numeric variables
- **Line Chart**: Time series or continuous ordered data
- **Heatmap**: Correlation matrix or any 2D matrix data
- **Pair Plot**: Explore relationships between multiple variables at once

### For Comparisons
- **Bar Chart**: Compare quantities across categories
- **Grouped Bar Chart**: Compare multiple metrics across categories
- **Stacked Bar Chart**: Show composition within categories
- **Radar Chart**: Compare multiple dimensions for few categories

### For Composition
- **Pie Chart**: Only when showing parts of a whole (max 5-6 slices)
- **Treemap**: Hierarchical composition with many categories
- **Stacked Area Chart**: Composition changes over time

## Design Principles

### Color Usage
- Use colorblind-friendly palettes (viridis, cividis)
- Sequential palettes for continuous data
- Diverging palettes for data with meaningful center point
- Categorical palettes for distinct groups (max 10 colors)

### Clarity
- Always include clear titles and axis labels
- Use appropriate font sizes (min 10pt for readability)
- Remove chart junk: unnecessary gridlines, borders, 3D effects
- Annotate key data points when relevant

### For Machine Learning Results
- **Confusion Matrix Heatmap**: Classification performance per class
- **ROC Curve**: Compare classifiers, show AUC
- **Feature Importance Bar Chart**: Horizontal bars, sorted by importance
- **Learning Curves**: Diagnose overfitting/underfitting
- **Residual Plot**: Validate regression assumptions

## Common Mistakes to Avoid
- Don't use pie charts for more than 5 categories
- Don't start bar chart Y-axis at non-zero (can be misleading)
- Don't use 3D charts - they distort perception
- Don't overcrowd with too many data points
- Always include units in axis labels
