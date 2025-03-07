import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset_path = "/Users/akankshabhimte/.cache/kagglehub/datasets/csafrit2/maternal-health-risk-data/versions/1/Maternal Health Risk Data Set.csv"
df = pd.read_csv(dataset_path)

# Check for unique values in RiskLevel
print("Unique Risk Levels (before processing):", df['RiskLevel'].unique())

# Strip any extra spaces
df['RiskLevel'] = df['RiskLevel'].str.strip()

# Convert RiskLevel from categorical to numerical
risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
df['RiskLevel'] = df['RiskLevel'].map(risk_mapping)

# Verify mapping
print("Unique Risk Levels (after mapping):", df['RiskLevel'].unique())

# Ensure no NaN values
if df['RiskLevel'].isna().sum() > 0:
    print("⚠️ Warning: Some RiskLevel values could not be mapped! Check dataset.")

# Split the dataset (80% training, 20% testing)
X = df.drop(columns=['RiskLevel'])  # Features
y = df['RiskLevel']  # Target label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n✅ Data Preprocessing Complete. Ready for Model Training!")


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters
print("\n🔥 Best Hyperparameters:", random_search.best_params_)

# Train the best model
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Tuned Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save the trained model
model_filename = "maternal_risk_model.pkl"
joblib.dump(best_rf, model_filename)
print(f"\n✅ Model saved as {model_filename}")

