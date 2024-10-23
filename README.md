# Income Classification Using Census Data

This project aims to uncover patterns and obtain insights from socio-economic factors, which are then used to classify an individual's income level. We utilized the **UCI Adult Census Income Dataset**, applying multiple machine learning techniques to build and compare models for binary income classification.

## Project Overview

The goal of this project is to classify whether a person earns more or less than $50,000 annually, based on attributes such as age, education level, occupation, etc. We implemented **four key machine learning models**—both using standard libraries and from-scratch implementations—to assess their effectiveness and performance.

### Models Implemented
1. **Random Forest**  
2. **Artificial Neural Networks (ANN)**  
3. **AdaBoost**  
4. **Decision Tree**

### Dataset
The **UCI Adult Census Income Dataset** was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/20/census+income). This dataset contains 48,842 records with 15 features representing various socio-economic and demographic data points.

## Data Preprocessing

Various data preprocessing techniques were applied to prepare the dataset for machine learning:

- **Handling Missing Data**: Missing values in columns like `native-country` and `workclass` were replaced with mode values. Special characters and NaN values were handled appropriately.
- **Duplicate Data Removal**: Fewer than 25 duplicate rows were removed as they were deemed inconsequential.
- **Categorical Variable Encoding**: Label Encoding, One-Hot-Encoding, and Frequency Encoding were used to convert categorical features into numerical representations.
- **Feature Scaling**: We applied both **Standard Scaler** and **MinMax Scaler** for feature scaling, essential for algorithms like Artificial Neural Networks.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was used to reduce the dimensionality of the dataset while preserving as much variance as possible.
- **Data Splitting**: The data was split into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets. We also employed **K-fold cross-validation** for model robustness testing.

## Models and Implementation

### Off-the-Shelf Implementations
We used the `scikit-learn` and `TensorFlow` libraries to implement the models with the following methods:
- **Decision Tree** and **Random Forest** (via `sklearn.ensemble`)
- **ANN** (via `TensorFlow` and `Keras`)
- **AdaBoost** (via `sklearn.ensemble`)

### Custom Implementations (From Scratch)
To deepen our understanding of these models, we implemented the following from scratch using `Numpy` and `Pandas`:
- **Random Forest**: Implemented from scratch, leveraging custom decision tree structures and bootstrapping techniques. This implementation was optimized for maximum efficiency.
- **Artificial Neural Networks (ANN)**: Built using custom classes for activation functions, loss functions, and optimization algorithms.
- **AdaBoost**: Created using custom decision stumps and iterative boosting techniques.
- **Decision Tree**: Constructed with recursive tree-building logic, including custom entropy and Gini index calculations.

### Hyperparameter Tuning
We employed **RandomizedSearchCV** and **GridSearchCV** to fine-tune the hyperparameters of the off-the-shelf models, ensuring maximum performance. The custom models used optimized parameters based on the results of this tuning process.

## Performance Evaluation

### Validation and Results
- Models were evaluated based on accuracy, precision, recall, and F1-score.
- We used **K-fold cross-validation** to further validate the robustness of the models.
- Comparison across four configurations:
  1. **Baseline off-the-shelf models** (e.g., Random Forest, Decision Tree, etc.)
  2. **Hyperparameter-tuned models** (using RandomizedSearchCV and GridSearchCV)
  3. **Custom implementations from scratch**
  4. **K-fold cross-validated custom models**

### Key Results
- **Random Forest**: Achieved an accuracy of **85.43%** (with hyperparameter tuning), while the custom implementation achieved **84.88%**, only a **0.8%** difference.
- **Artificial Neural Networks**: Slight performance improvement after tuning with a **85.30%** accuracy.
- **AdaBoost**: The largest difference was observed here, where the custom AdaBoost had a **7%** lower accuracy compared to the baseline.
- **Decision Tree**: Baseline accuracy of **81.36%**, improved to **85.93%** with tuning.

### Summary of Model Performance

| Model                  | Accuracy  | Precision | Recall | F1 Score |
|------------------------|-----------|-----------|--------|----------|
| **Random Forest**       | 85.43%    | 76.43%    | 56.62% | 67%      |
| **Artificial Neural Network** | 85.30%    | 79.86%    | 51.60% | 62.69%   |
| **AdaBoost**            | 86%       | 75%       | 61%    | 67%      |
| **Decision Tree**       | 85.93%    | 78.97%    | 56.16% | 65.64%   |
