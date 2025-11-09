# Experiment 2: Data Preprocessing on Real-World Dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load dataset
url = "C:\Titanic-Dataset.csv"
data = pd.read_csv(url)

print("Original Data Shape:", data.shape)
print("Missing Values Before Processing:\n", data.isnull().sum())

# Step 2: Handle missing values — FIXED (no inplace=True)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop irrelevant or highly missing columns
data = data.drop(columns=['Cabin', 'Name', 'Ticket'])

# Step 3: Handle outliers using IQR method for 'Fare'
Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
data = data[(data['Fare'] >= lower) & (data['Fare'] <= upper)]

# Step 4: Encode categorical variables
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])
data['Embarked'] = encoder.fit_transform(data['Embarked'])

# Step 5: Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Age', 'Fare', 'SibSp', 'Parch']])
scaled_df = pd.DataFrame(scaled_features, columns=['Age', 'Fare', 'SibSp', 'Parch'])

# Step 6: Replace original numeric columns with scaled values
data[['Age', 'Fare', 'SibSp', 'Parch']] = scaled_df

# Step 7: Display cleaned and preprocessed dataset
print("\nData After Preprocessing:\n", data.head())
print("\nMissing Values After Processing:\n", data.isnull().sum())
print("\nFinal Data Shape:", data.shape)
