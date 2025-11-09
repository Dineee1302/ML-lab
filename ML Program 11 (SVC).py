# Support Vector Classification (SVC) with Linear Kernel

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
url = "C:\diabetes.csv"  # Replace with your path
data = pd.read_csv(url)

# Step 2: Inspect dataset
print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# Step 3: Prepare features and target
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']               # Target (1 = diabetes, 0 = no diabetes)

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Feature scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train SVC with linear kernel
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = svc.predict(X_test_scaled)

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
