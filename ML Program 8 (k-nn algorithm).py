# k-NN Classification with automatic k selection on SMS Spam dataset

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
data = pd.read_csv("C:\spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]  # 'v1' = label, 'v2' = message
data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Encode labels: 'ham' = 0, 'spam' = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 2: Prepare features and target
X_text = data['message']
y = data['label']

# Step 3: Convert text to numeric features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(X_text).toarray()

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Use GridSearchCV to find the best k
param_grid = {'n_neighbors': list(range(1, 21))}  # try k from 1 to 20
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best k value
best_k = grid_search.best_params_['n_neighbors']
print("Best k found by cross-validation:", best_k)

# Step 7: Train k-NN with best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

# Step 8: Make predictions
y_pred = knn_best.predict(X_test_scaled)

# Step 9: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
