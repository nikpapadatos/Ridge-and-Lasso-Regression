# Shrinkage Methods: Lasso and Ridge Regression – Theory and Comparison

## Project Overview
This project explores the application of Lasso and Ridge regression, two widely-used shrinkage methods. Both techniques address overfitting by penalizing large coefficients. The goal is to compare their performance in feature selection and coefficient shrinkage using a dataset containing various credit-related factors.

## Key Topics Covered:
- Ridge Regression (L2 regularization): Shrinks all coefficients towards zero without eliminating any, making it useful when all features contribute to the prediction.
- Lasso Regression (L1 regularization): Can shrink some coefficients to zero, which is beneficial for feature selection.
- Comparison: A side-by-side comparison of how Ridge and Lasso affect coefficient estimates and their suitability for different use cases.

## Dataset
The dataset used for this analysis is the **Credit dataset**, which contains various credit-related variables like `Income`, `Limit`, `Rating`, and `Balance` (the target variable). The aim is to predict the `Balance` using the other features.

## Project Structure

### 1. Data Loading and Preprocessing
- **Libraries**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`
- **Steps**:
  - Load the dataset and display its summary statistics.
  - Visualize the relationships between variables using a pairplot.
  - Preprocess the dataset by one-hot encoding categorical variables and standardizing the features.

### 2. Ridge Regression
- **Theory**: Ridge Regression applies L2 regularization, adding a penalty term proportional to the square of the coefficients. This helps shrink the coefficients while retaining all variables.
- **Implementation**: Ridge regression is applied to the dataset using a range of alpha values (regularization strength). The solution path for Ridge is visualized, showing how coefficients shrink as alpha increases.

### 3. Lasso Regression
- **Theory**: Lasso Regression applies L1 regularization, which can set some coefficients exactly to zero, effectively performing feature selection.
- **Implementation**: Similar to Ridge, Lasso regression is applied with varying alpha values. The Lasso solution path is visualized, demonstrating how coefficients behave as the regularization strength increases.

### 4. Comparison of Ridge and Lasso
- A side-by-side comparison of the coefficients obtained using Ridge and Lasso at a fixed alpha value is performed. The comparison highlights:
  - Ridge shrinks coefficients uniformly, while Lasso sets some coefficients to zero.
  - Lasso performs feature selection, making it a good choice when interpretability and selecting important features are priorities.
 
### 5. Bias-Variance Tradeoff

- Ridge: By shrinking coefficients, Ridge reduces model variance at the cost of introducing some bias. It is especially useful when preventing overfitting is important.
- Lasso: Lasso also introduces bias by shrinking coefficients, but because it can set coefficients to zero, it reduces model complexity more aggressively than Ridge. This makes Lasso     better suited for high-dimensional datasets with many irrelevant features.

### 6. Observations and Conclusions
- Ridge regression tends to shrink all coefficients gradually, with no coefficients being eliminated, even with high alpha values.
- Lasso regression sets some coefficients to zero, effectively removing less important features as alpha increases.
- **Key Insight**: Lasso is useful for feature selection, while Ridge is better suited when retaining all features is more important, but controlling their magnitude.
