import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define directories
data_dir = r"C:\Users\ri\OneDrive\ai project\model\Titanic Smart Features Model Blending\data"
image_dir = r"C:\Users\ri\OneDrive\ai project\model\Titanic Smart Features Model Blending\images"

# Ensure output directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Step 1: Feature Engineering
print("Loading and performing feature engineering...")
df = pd.read_csv(os.path.join(data_dir, 'train_scaled.csv'))

# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# Save engineered dataset
engineered_path = os.path.join(data_dir, 'train_engineered.csv')
df.to_csv(engineered_path, index=False)
print(f"Engineered dataset saved as '{engineered_path}'")

# Prepare features and target
X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived']
print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Step 2: Train Individual Models
# Logistic Regression
print("\nTraining Logistic Regression model...")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {acc_logreg:.4f}")
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg, target_names=['Not Survived', 'Survived']))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))

# Random Forest
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Not Survived', 'Survived']))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# XGBoost
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb:.4f}")
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Not Survived', 'Survived']))
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Step 3: Build Stacking Ensemble
print("\nTraining Stacking Ensemble model...")
estimators = [
    ('logreg', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', xgb.XGBClassifier(eval_metric='logloss', random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_test)

# Step 4: Evaluate Ensemble
acc_stack = accuracy_score(y_test, y_pred_stack)
print(f"\nStacking Ensemble Accuracy: {acc_stack:.4f}")
print("Stacking Ensemble Classification Report:")
print(classification_report(y_test, y_pred_stack, target_names=['Not Survived', 'Survived']))
cm_stack = confusion_matrix(y_test, y_pred_stack)
print("Stacking Ensemble Confusion Matrix:")
print(cm_stack)

# Compare ensemble to individual models
print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {acc_logreg:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print(f"XGBoost Accuracy: {acc_xgb:.4f}")
print(f"Stacking Ensemble Accuracy: {acc_stack:.4f}")
if acc_stack > max(acc_logreg, acc_rf, acc_xgb):
    print("The ensemble model outperforms all individual models.")
elif acc_stack == max(acc_logreg, acc_rf, acc_xgb):
    print("The ensemble model performs as well as the best individual model.")
else:
    print("The ensemble model does not outperform the best individual model.")

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Stacking Ensemble Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
cm_path = os.path.join(image_dir, 'confusion_matrix_ensemble.png')
plt.savefig(cm_path)
plt.close()
print(f"Ensemble confusion matrix saved as '{cm_path}'")

# Save predictions
predictions_df = pd.DataFrame({
    'PassengerId': df.loc[X_test.index, 'PassengerId'],
    'Survived': y_pred_stack
})
predictions_path = os.path.join(data_dir, 'ensemble_predictions.csv')
predictions_df.to_csv(predictions_path, index=False)
print(f"Ensemble predictions saved as '{predictions_path}'")