# Logistic Regression for Diabetes Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
# Replace the path with your local CSV file
# Example: "diabetes.csv" from Kaggle or UCI
url = "C:\diabetes.csv"
data = pd.read_csv(url)

# Step 2: Inspect the dataset
print("Dataset shape:", data.shape)
print("Columns:", data.columns)
print("First 5 rows:\n", data.head())

# Step 3: Prepare features and target
# Assume the dataset has a column 'Outcome' as target (1 = diabetes, 0 = no diabetes)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = logreg.predict(X_test_scaled)

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
