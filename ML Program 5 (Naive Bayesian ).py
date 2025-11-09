import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
data = pd.read_csv("C:\Cancer_Data.csv")

# Step 2: Inspect the dataset
print("Shape of DataFrame:", data.shape)
print("Columns in data:", data.columns)

data = data.drop(['id', 'Unnamed: 32'], axis=1)
data = data.dropna()

# Step 3: Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Step 4: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Step 3: Clean dataset (handle missing values)


# Step 8: Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)

# Step 10: Evaluate
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
