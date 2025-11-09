import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Load the Wine Quality Dataset
url = "C:\winequality-white.csv"
data = pd.read_csv(url, sep=";")

# Step 2: Inspect the dataset
print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# Step 3: Preprocess the data
X = data.drop('quality', axis=1)  # Features
y = data['quality']  # Target (not used in unsupervised learning)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply Gaussian Mixture Model (GMM)
n_clusters = 6
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# Step 5: Predict cluster labels
labels = gmm.predict(X_scaled)
data['Cluster'] = labels

# Step 6: Evaluate clustering quality
sil_score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", sil_score)

# Step 7: Visualize clusters (first two features)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.title('Wine Clustering using GMM')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.colorbar(label='Cluster')
plt.show()
