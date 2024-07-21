import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import numpy as np

# Load dataset
df = pd.read_csv('NEW_FINAL_DATASET (2).CSV')

# Handle class imbalance by upsampling minority class
df_majority = df[df.Attrition == 0]
df_minority = df[df.Attrition == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Define feature columns and target column
X = df_upsampled.drop('Attrition', axis=1)
y = df_upsampled['Attrition']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define parameter grids for different models
param_grids = {
    'svm': {
        'classifier': [SVC()],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__C': np.logspace(-3, 2, 6),
        'classifier__gamma': ['scale', 'auto']
    },
    'logistic_regression': {
        'classifier': [LogisticRegression()],
        'classifier__C': np.logspace(-3, 2, 6),
        'classifier__solver': ['liblinear', 'lbfgs']
    },
    'decision_tree': {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'random_forest': {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
}

# Compare models and hypertune
best_model = None
best_score = 0
best_params = None

for model_name, param_grid in param_grids.items():
    print(f"Evaluating model: {model_name}")
    
    # Define a pipeline for the current model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', param_grid['classifier'][0])
    ])
    
    # Use RandomizedSearchCV instead of GridSearchCV
    randomized_search = RandomizedSearchCV(pipeline, param_grid, n_iter=20, cv=StratifiedKFold(n_splits=5), scoring='accuracy', random_state=42)
    randomized_search.fit(X_train, y_train)
    
    y_pred = randomized_search.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(report)
    print(f"Confusion Matrix for {model_name}:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Best Params for {model_name}: {randomized_search.best_params_}")
    print(f"Best Cross-Validation Score for {model_name}: {randomized_search.best_score_}\n")
    
    if randomized_search.best_score_ > best_score:
        best_score = randomized_search.best_score_
        best_model = randomized_search.best_estimator_
        best_params = randomized_search.best_params_

# Print best model and score
print(f"Best Model: {best_model}")
print(f"Best Score: {best_score}")
print(f"Best Params: {best_params}")

# Evaluate on test data
y_pred = best_model.predict(X_test)
print("Classification report for the best model:")
print(classification_report(y_test, y_pred))
print(f"Confusion Matrix for the best model:\n{confusion_matrix(y_test, y_pred)}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')

