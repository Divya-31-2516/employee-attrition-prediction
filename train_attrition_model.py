import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Load the dataset
file_path = 'F:/PROHJECT/employee_data.csv'  # Correct path to your dataset
df = pd.read_csv(file_path)

# Check the first few rows to understand the dataset structure
print(df.head())

# Preprocessing
# Assuming 'Attrition' is the target variable and we need to handle categorical features
# For simplicity, let's use 'Attrition' as the target and the rest as features.

# Drop any columns that may not be useful (e.g., EmployeeNumber, Name, etc.)
df = df.drop(columns=['EmployeeNumber', 'Name'], axis=1)

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Convert categorical columns to numerical using pd.get_dummies or LabelEncoder
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, drop_first=True)

# Define the target and features
X = df.drop('Attrition_Yes', axis=1)  # Assuming 'Attrition_Yes' is the binary target variable
y = df['Attrition_Yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with SMOTE for handling class imbalance
pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto')),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define grid search for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
import joblib
joblib.dump(best_model, 'attrition_model.pkl')
print("Model saved as attrition_model.pkl")
