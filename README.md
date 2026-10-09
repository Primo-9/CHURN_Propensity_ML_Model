# Data Source

The data in this project was obtained from IBM Cognos Analytics.

Source: https://www.ibm.com/docs/en/cognos-analytics/12.1.x?topic=samples-telco-customer-churn

This sample data module tracks a fictional telco company's customer churn based on various factors.

The target variable is "Churn Value"

# Customer Churn Prediction – End-to-End Machine Learning Pipeline 🧠

Customer churn is a critical business problem in subscription-based industries.
Acquiring new customers is significantly more expensive than retaining existing ones, so early identification of churn risk is essential.

This project builds an end-to-end machine learning pipeline to predict customer churn using behavioral, contract, and service usage data.

The goal is not only high accuracy, but high recall and probability-based risk scoring, enabling proactive customer retention strategies.

# Objectives
* Predict which customers are likely to churn.

* Maximize recall (catch as many churners as possible), while trying to achieve a reasonable precision.

* Produce probabilities, not just hard classifications

* Build a robust, production-ready pipeline.

* Compare multiple models and evaluate stacking performance

# Tools & Technologies

* Python
  - scikit-learn
  - imblearn
  - XGBoost
  - pandas  / numpy
  - matplotlib / seaborn

# Methodology

1. EDA
2. Data Cleaning
3. Feature Engineering
4. Handling Class Imbalance
5. Model Selection and Hyperparameter Tuning (RF, XGB, LR, Stack)
6. Result Visualization and Analysis

# Model Performance

| Model | ROC-AUC (Test) | ROC-AUC (5-Fold CV) | PR-AUC (Test) | Test-CV Gap | Ranking | Business Impact |
|-------|----------------|---------------------|---------------|-------------|---------|-----------------|
| **Stacking Ensemble** | **0.8847** 🥇 | **0.8587** | **0.6861** 🥇 | 2.60% | **1st** | Identifies 89% of churners with 53% precision |
| Logistic Regression | 0.8831 🥈 | 0.8595 | 0.6824 🥈 | 2.36% | 2nd | Nearly identical performance, 10x faster |
| XGBoost | 0.8808 | 0.8354 | 0.6419 | 4.54% | 3rd | Good but unstable (4.5% test-CV gap) |
| Random Forest | 0.8763 | 0.8386 | 0.6405 | 3.77% | 4th | Lowest PR-AUC, prone to false positives |
| *Random Baseline* | *0.5000* | - | *0.2650* | - | - | - |

**Key Metrics:**
- **ROC-AUC:** Measures overall discrimination ability (0.5 = random, 1.0 = perfect)
- **PR-AUC:** Precision-Recall AUC - more informative for imbalanced datasets (baseline = 26.5% churn rate)
- **Test-CV Gap:** Difference between test and cross-validation scores (lower = better generalization)

**Winner:** Stacking Ensemble achieves the best performance on both metrics while maintaining excellent stability (2.60% gap).

### Why These Metrics Matter

**ROC-AUC (Receiver Operating Characteristic)**
- Measures how well the model distinguishes between churners and non-churners
- Best model: **88.47%** (excellent discrimination)

<img width="559" height="443" alt="image" src="https://github.com/user-attachments/assets/7b58920a-6fa4-49ef-b5c2-158a75325c73" />

**PR-AVG (Precision-Recall)**
- More reliable for imbalanced data (26.5% churn rate)
- Shows real-world performance: **68.61%** average precision across all recall levels
- **2.59x better** than random predictions

<img width="557" height="441" alt="image" src="https://github.com/user-attachments/assets/f9c0a9bc-eb4a-45f6-9a6f-2f46c1a26742" />


### Model Selection

**For Production:** Logistic Regression (0.6824 PR-AUC) 🏆
- Only 0.54% worse than Stacking
- 10x faster inference time
- Fully interpretable for business stakeholders

**Technical Achievement:** Stacking Ensemble (0.6861 PR-AUC)
- State-of-the-art performance
- Best for high-stakes predictions

# Discussion

This project successfully developed a machine learning solution to predict customer churn in the telecommunications industry, achieving 88.47% ROC-AUC and 68.61% PR-AUC using a Stacking Ensemble approach. More importantly, the model demonstrates practical business value by identifying 89% of potential churners while maintaining operational feasibility.
This model was trained for predicting churn, the churn rate for this dataset was 26.5%. Since churn is an imbalanced class, accuracy is not a realistic metric to determine the churn rate, for this reason, recall,  precision and PR-AVG were the main focus.

<img width="542" height="417" alt="image" src="https://github.com/user-attachments/assets/65cc8b68-7fd4-4d80-9127-b73386f4f518" />


**Stacking Ensemble** vs **Logistic Regression**: +0.16% ROC-AUC, +0.37% PR-AUC.
**Critical Observation**: The gap between Stacking and Logistic Regression is remarkably small (0.37% in PR-AUC), while the gap to tree-based models is significantly larger (4.42-4.56%). This reveals an important insight about the nature of this dataset.
The strong performance of LR (a simple linear model) suggests that this datasets relationship between features and churn is predominantly linear, hence the performance of LR  ≈ StackEnsemble. Realistically LR is a better choice in general because it is 10x faster and less complex while performing optimally at predicting churners.

XGB did not perform as well because the dataset is small (7043, 33), and XGB excels with big datasets.

RandomForest also performed poorly, RF precision drops rapidly as we increase recall, this manifests as low PR-AUC.

One of the most critical findings is that the optimal threshold is NOT 0.5 (the default). At threshold = 0.2:
- Recall: 88% (catch 369 of 374 churners)
- Precision: 53% (369 correct of 973 predictions)
It is much more costly to lose a customer, than to falsely contact a not-leaving customer.
So business wise it is feasible trading the precision for a higher recall, thus saving a lot of $$$.

<img width="578" height="466" alt="image" src="https://github.com/user-attachments/assets/4b2fdae9-eee7-4f6f-9f7e-749520d05805" />


# Conclusions

1. Simpler is often Better
2. PR-AUC > ROC-AUC for Imbalanced Data
3. Feature Engineering > Model Complexity
4. Threshold Optimization is Critical

****The key finding****: A well-engineered Logistic Regression (0.6824 PR-AUC) delivers 99.5% of a complex Stacking Ensemble's performance (0.6861 PR-AUC) while being 10x faster and fully interpretable.

This demonstrates the principle that in production ML, the best model is not always the most complex one—it's the one that best balances performance, interpretability, and operational feasibility.




