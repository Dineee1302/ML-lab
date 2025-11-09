# Experiment 3: Feature Subset Selection Techniques

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress all harmless warnings
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
url = "C:\Titanic-Dataset.csv"
data = pd.read_csv(url).copy()   # ensure fresh copy to avoid chained assignment issues

# Step 2: Preprocess dataset (safe assignments)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data = data.drop(columns=['Cabin', 'Name', 'Ticket'])

# Encode categorical variables
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])
data['Embarked'] = encoder.fit_transform(data['Embarked'])

# Step 3: Split features and target
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
y = data['Survived']

# Step 4: Filter Method (Correlation-based)
correlation = X.corr(numeric_only=True)  # explicit for sklearn >=1.5 compatibility
print("Correlation Matrix:\n", correlation)
print("\nHighly Correlated Features (>|0.8|):")
print(correlation[(correlation > 0.8) | (correlation < -0.8)])

# Step 5: Wrapper Method (Recursive Feature Elimination - RFE)
model = LogisticRegression(max_iter=2000, solver='lbfgs')
rfe = RFE(model, n_features_to_select=4)
fit = rfe.fit(X, y)

print("\nSelected Features using RFE:")
for i, col in enumerate(X.columns):
    if fit.support_[i]:
        print(f"- {col}")

# Step 6: Embedded Method (Feature Importance using Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nFeature Importance (Random Forest):\n", importances)

# Step 7: Display top features
top_features = importances.head(4).index.tolist()
print("\nTop 4 Important Features (Final Selected):", top_features)
