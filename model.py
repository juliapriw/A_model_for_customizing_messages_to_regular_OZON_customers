import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = 'data.xlsx' 
data = pd.read_excel(file_path)

# Data cleaning and conversion
data['пол'] = data['пол'].astype(str).fillna('Unknown')
data['доминирующий цвет'] = data['доминирующий цвет'].astype(str).fillna('Unknown')
data['доминирующая категория товара'] = data['доминирующая категория товара'].astype(str)

# Convert numerical data to floats explicitly
numerical_cols = ['возраст', 'Средний чек']
for col in numerical_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())

# Define preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), ['пол', 'доминирующий цвет'])
    ])

# Creating a pipeline with RandomForest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Setup the grid search
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Applying grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(data[numerical_cols + ['пол', 'доминирующий цвет']], data['доминирующая категория товара'])

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(data[numerical_cols + ['пол', 'доминирующий цвет']])
y_true = data['доминирующая категория товара']

print("Final Model Accuracy:", accuracy_score(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_true, y_pred, labels=best_model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model, data[numerical_cols + ['пол', 'доминирующий цвет']], data['доминирующая категория товара'], 
    cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 8))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.title("Learning Curve")
plt.xlabel("Training Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.show()
