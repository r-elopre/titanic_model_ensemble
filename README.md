# Titanic Survival Prediction with Feature Engineering and Model Stacking
<p align="center">
  <a href="https://youtu.be/lbJZueex-NQ?si=TCNnxzdHKLXPZLbp">
    <img src="https://img.youtube.com/vi/lbJZueex-NQ/maxresdefault.jpg" alt="Titanic Stacking Ensemble Model Video">
  </a>
  <br/>
  <em>Click the thumbnail to watch on YouTube</em>
</p>

## Project Overview

This project aims to predict passenger survival on the Titanic using advanced machine learning techniques, specifically feature engineering and model stacking. The goal is to enhance predictive performance by creating meaningful features from the dataset and combining the strengths of multiple models (Logistic Regression, Random Forest, and XGBoost) into a stacking ensemble. The project fulfills the requirements of the "Improve Titanic Predictions with Smart Features + Model Blending" activity, producing a feature-engineered dataset, individual model evaluations, an ensemble model, and comprehensive performance metrics.

## Dataset Description

### Input Data
- **File**: `data/train_scaled.csv`
- **Source**: Preprocessed Titanic dataset, assumed to be derived from the Kaggle Titanic dataset.
- **Columns** (18 total):
  - **Target**: `Survived` (0 = Not Survived, 1 = Survived)
  - **Identifier**: `PassengerId`
  - **Features**:
    - Numerical: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
    - Categorical (encoded as binary): `Sex_male`, `Embarked_q`, `Embarked_s`, `Deck_B`, `Deck_C`, `Deck_D`, `Deck_E`, `Deck_F`, `Deck_G`, `Deck_T`, `Deck_Unknown`
- **Shape**: 891 rows, 18 columns (before feature engineering)
- **Note**: The `Name` column, typically used for extracting titles (e.g., Mr, Mrs), was not available, so alternative features were created.

### Feature Engineering
Three new features were engineered to capture additional patterns in the data:
1. **FamilySize**: Sum of `SibSp` (siblings/spouses aboard), `Parch` (parents/children aboard), and 1 (the passenger themselves). This represents the total family size onboard.
   - Formula: `FamilySize = SibSp + Parch + 1`
2. **IsAlone**: Binary indicator of whether the passenger is traveling alone (1 if `FamilySize == 1`, else 0).
   - Rationale: Passengers traveling alone may have different survival probabilities due to social or logistical factors.
3. **FarePerPerson**: Fare divided by `FamilySize`, representing the per-person cost of the ticket.
   - Formula: `FarePerPerson = Fare / FamilySize`
   - Rationale: Accounts for shared tickets among families, potentially correlating with socioeconomic status.

- **Output**: The engineered dataset (891 rows, 19 columns, including the 3 new features) is saved as `data/train_engineered.csv`.
- **Impact**: These features enhance the models' ability to capture relationships between family dynamics, socioeconomic factors, and survival outcomes.

## Methodology

### Data Preprocessing
- **Loading**: The dataset is loaded from `C:\Users\ri\OneDrive\ai project\model\Titanic Smart Features Model Blending\data\train_scaled.csv`.
- **Train-Test Split**: 80/20 split (712 training samples, 179 test samples) with `random_state=42` for reproducibility.
- **Features**: All columns except `PassengerId` and `Survived` are used as features (16 original + 3 engineered = 19 features).
- **Target**: `Survived` (binary classification: 0 = Not Survived, 1 = Survived).

### Models
Four models were trained and evaluated:
1. **Logistic Regression**:
   - Parameters: `max_iter=1000`, `random_state=42`
   - Description: A linear model serving as a baseline, effective for binary classification with interpretable coefficients.
2. **Random Forest**:
   - Parameters: Default settings, `random_state=42`
   - Description: An ensemble of decision trees, robust to overfitting and capable of capturing non-linear relationships.
3. **XGBoost**:
   - Parameters: `eval_metric='logloss'`, `random_state=42`
   - Description: A gradient boosting model, optimized for performance and handling complex feature interactions.
4. **Stacking Ensemble**:
   - Configuration: `StackingClassifier` combining Logistic Regression, Random Forest, and XGBoost, with Logistic Regression as the final estimator.
   - Description: Combines predictions from the three base models to leverage their complementary strengths, potentially improving overall performance.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions.
- **Classification Report**: Precision, recall, F1-score for each class (`Not Survived`, `Survived`), plus macro and weighted averages.
- **Confusion Matrix**: Visualized using a heatmap to show true positives, true negatives, false positives, and false negatives.
- **Comparison**: The ensemble's accuracy is compared to individual models to assess improvement.

### Outputs
- **Engineered Dataset**: `data/train_engineered.csv`
- **Predictions**: `data/ensemble_predictions.csv` (contains `PassengerId` and predicted `Survived` for the test set)
- **Visualization**: Confusion matrix for the ensemble saved as `images/confusion_matrix_ensemble.png`
- **Console Output**: Accuracy, classification reports, and confusion matrices for all models.

## Results

The script was executed on August 4, 2025, at 09:19 AM PST, producing the following results:

### Dataset
- **Features Shape**: (891, 19)
- **Target Shape**: (891,)
- **Training Set**: 712 samples
- **Test Set**: 179 samples

### Model Performance
1. **Logistic Regression**:
   - **Accuracy**: 0.8212
   - **Classification Report**:
     ```
                  precision    recall  f1-score   support
     Not Survived       0.84      0.86      0.85       105
         Survived       0.79      0.77      0.78        74
         accuracy                           0.82       179
        macro avg       0.82      0.81      0.81       179
     weighted avg       0.82      0.82      0.82       179
     ```
   - **Confusion Matrix**:
     ```
     [[90 15]
      [17 57]]
     ```
   - **Analysis**: Strong performance with balanced precision and recall, slightly better at predicting `Not Survived`.

2. **Random Forest**:
   - **Accuracy**: 0.8101
   - **Classification Report**:
     ```
                  precision    recall  f1-score   support
     Not Survived       0.84      0.83      0.84       105
         Survived       0.76      0.78      0.77        74
         accuracy                           0.81       179
        macro avg       0.80      0.81      0.80       179
     weighted avg       0.81      0.81      0.81       179
     ```
   - **Confusion Matrix**:
     ```
     [[87 18]
      [16 58]]
     ```
   - **Analysis**: Slightly lower accuracy than Logistic Regression, with more false positives (`Not Survived` predicted as `Survived`).

3. **XGBoost**:
   - **Accuracy**: 0.8268
   - **Classification Report**:
     ```
                  precision    recall  f1-score   support
     Not Survived       0.84      0.87      0.85       105
         Survived       0.80      0.77      0.79        74
         accuracy                           0.83       179
        macro avg       0.82      0.82      0.82       179
     weighted avg       0.83      0.83      0.83       179
     ```
   - **Confusion Matrix**:
     ```
     [[91 14]
      [17 57]]
     ```
   - **Analysis**: Outperforms Random Forest and slightly edges out Logistic Regression, with strong recall for `Not Survived`.

4. **Stacking Ensemble**:
   - **Accuracy**: 0.8324
   - **Classification Report**:
     ```
                  precision    recall  f1-score   support
     Not Survived       0.84      0.89      0.86       105
         Survived       0.82      0.76      0.79        74
         accuracy                           0.83       179
        macro avg       0.83      0.82      0.82       179
     weighted avg       0.83      0.83      0.83       179
     ```
   - **Confusion Matrix**:
     ```
     [[93 12]
      [18 56]]
     ```
   - **Analysis**: Achieves the highest accuracy (0.8324), with the best recall for `Not Survived` (0.89) and strong overall performance. The ensemble reduces false positives compared to Random Forest and false negatives compared to Logistic Regression.

### Model Comparison
- **Accuracies**:
  - Logistic Regression: 0.8212
  - Random Forest: 0.8101
  - XGBoost: 0.8268
  - Stacking Ensemble: 0.8324
- **Conclusion**: The stacking ensemble outperforms all individual models, improving accuracy by 0.0056 over XGBoost, 0.0112 over Logistic Regression, and 0.0223 over Random Forest. The ensemble leverages the strengths of all three models, particularly improving recall for `Not Survived` (0.89 vs. 0.86–0.87 for individual models).

### Outputs
- **Engineered Dataset**: `C:\Users\ri\OneDrive\ai project\model\Titanic Smart Features Model Blending\data\train_engineered.csv`
- **Predictions**: `C:\Users\ri\OneDrive\ai project\model\Titanic Smart Features Model Blending\data\ensemble_predictions.csv`
- **Confusion Matrix Plot**: `C:\Users\ri\OneDrive\ai project\model\Titanic Smart Features Model Blending\images\confusion_matrix_ensemble.png`

## Implementation Details

### Script: `titanic_model_ensemble.py`
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation
  - `sklearn`: Train-test split, Logistic Regression, Random Forest, StackingClassifier, metrics
  - `xgboost`: XGBoost classifier
  - `seaborn`, `matplotlib`: Visualization
  - `os`: File path handling
- **Steps**:
  1. **Feature Engineering**: Creates `FamilySize`, `IsAlone`, `FarePerPerson` and saves the dataset.
  2. **Train-Test Split**: 80/20 split with `random_state=42`.
  3. **Model Training**: Trains Logistic Regression, Random Forest, and XGBoost on the engineered dataset.
  4. **Stacking Ensemble**: Combines the three models using `StackingClassifier` with Logistic Regression as the final estimator.
  5. **Evaluation**: Prints accuracy, classification report, and confusion matrix for each model and the ensemble. Compares ensemble performance to individual models.
  6. **Output Saving**: Saves the engineered dataset, ensemble predictions, and ensemble confusion matrix plot to specified directories.
- **Path Handling**: Uses absolute paths with `os.path.join` for robustness, creating `data` and `images` directories if they don’t exist.
- **Reproducibility**: Sets `np.random.seed(42)` and `random_state=42` in all models.

### File Structure
```
Titanic Smart Features Model Blending/
├── data/
│   ├── train_scaled.csv              # Input dataset
│   ├── train_engineered.csv          # Engineered dataset
│   ├── ensemble_predictions.csv      # Ensemble predictions
├── images/
│   ├── confusion_matrix_ensemble.png # Ensemble confusion matrix plot
├── titanic_model_ensemble.py         # Main script
├── README.md                         # Project documentation
```

## Reproducibility Instructions

### Prerequisites
- Python 3.8+
- Libraries: Install via `pip install pandas numpy scikit-learn xgboost seaborn matplotlib`
- Dataset: Ensure `train_scaled.csv` is in `C:\Users\ri\OneDrive\ai project\model\Titanic Smart Features Model Blending\data`

### Steps to Run
1. **Set Up Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install pandas numpy scikit-learn xgboost seaborn matplotlib
   ```
2. **Place Input Data**: Ensure `train_scaled.csv` is in the `data` directory.
3. **Run Script**:
   ```bash
   python titanic_model_ensemble.py
   ```
4. **Expected Outputs**:
   - Console: Accuracy, classification reports, confusion matrices, and model comparison.
   - Files: `train_engineered.csv`, `ensemble_predictions.csv` in `data/`, and `confusion_matrix_ensemble.png` in `images/`.

### Notes
- The script assumes `train_scaled.csv` has the same structure as described (18 columns, including `PassengerId`, `Survived`, and 16 features).
- If the `Name` column were available, a `Title` feature could be extracted using regex (e.g., `df['Name'].str.extract(' ([A-Za-z]+)\.')`), potentially improving performance.
- The stacking ensemble’s performance depends on the diversity of base models; further tuning (e.g., hyperparameter optimization) could enhance results.

## Analysis and Insights

- **Feature Engineering Impact**: The addition of `FamilySize`, `IsAlone`, and `FarePerPerson` likely improved model performance by capturing social and economic factors. For example, `IsAlone` may correlate with survival due to differences in evacuation behavior, and `FarePerPerson` adjusts for shared tickets, refining the socioeconomic signal of `Fare`.
- **Model Strengths**:
  - **Logistic Regression**: Reliable baseline, excels in linearly separable data.
  - **Random Forest**: Captures non-linear patterns but had the lowest accuracy (0.8101), possibly due to overfitting or default parameters.
  - **XGBoost**: Strong performance (0.8268) due to gradient boosting, handling feature interactions well.
  - **Stacking Ensemble**: Best performer (0.8324), leveraging complementary strengths, particularly improving recall for `Not Survived` (0.89).
- **Ensemble Advantage**: The stacking ensemble’s slight accuracy improvement (0.8324 vs. 0.8268 for XGBoost) suggests it effectively combines model predictions, reducing errors (fewer false positives than Random Forest, fewer false negatives than Logistic Regression).
- **Limitations**:
  - Lack of `Name` column prevented `Title` extraction, which could have added predictive power (e.g., distinguishing officers or nobility).
  - Default parameters were used for Random Forest and XGBoost; hyperparameter tuning (as in your `titanic_model_xgboost_tuned.py`) could further improve results.
  - The dataset’s small size (891 samples) limits model complexity; larger datasets might benefit more from ensemble methods.

## Future Improvements
1. **Hyperparameter Tuning**: Use GridSearchCV for Random Forest and XGBoost to optimize parameters (e.g., `n_estimators`, `max_depth`, `learning_rate`).
2. **Additional Features**: If raw data with `Name` or `Ticket` is available, extract `Title` or ticket-based features (e.g., ticket group size).
3. **Alternative Ensembles**: Experiment with VotingClassifier (soft or hard voting) or blending instead of stacking.
4. **Feature Importance**: Analyze feature importance (e.g., using XGBoost’s `feature_importances_`) to identify key predictors.
5. **Cross-Validation**: Implement k-fold cross-validation for more robust performance estimates.

## Conclusion

This project successfully implemented feature engineering and model stacking to predict Titanic survival, achieving a test accuracy of 0.8324 with the stacking ensemble, outperforming individual models (Logistic Regression: 0.8212, Random Forest: 0.8101, XGBoost: 0.8268). The engineered features (`FamilySize`, `IsAlone`, `FarePerPerson`) enhanced the models’ ability to capture meaningful patterns, and the stacking ensemble effectively combined model strengths. All required outputs were generated, and the project is fully reproducible with the provided script and instructions.