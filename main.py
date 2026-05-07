import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
import joblib

warnings.filterwarnings('ignore')  # suppress sklearn KMeans warnings on Windows

# Load the Mall Customers dataset
df = pd.read_csv('Mall_Customers.csv')

# Display the first few rows to verify the data was loaded correctly
print("First 5 rows of the dataset:")
print(df.head())

# Display some basic information about the dataset
print("\n--- Dataset Info ---")
df.info()

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Display statistical summary of the dataset
print("\n--- Statistical Summary ---")
print(df.describe())

# Display the distribution of Gender (labeled as 'Genre' in this dataset)
print("\n--- Gender Distribution ---")
print(df['Genre'].value_counts())

# Create a figure with two subplots for gender visualization
plt.figure(num='Customer Demographics: Gender Distribution', figsize=(12, 5))

# Subplot 1: Barplot for gender distribution
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Genre', hue='Genre', palette='Set2', legend=False)
plt.title('Gender Distribution - Barplot')
plt.xlabel('Gender')
plt.ylabel('Count')

# Subplot 2: Pie chart for gender distribution
plt.subplot(1, 2, 2)
gender_counts = df['Genre'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'), startangle=90)
plt.title('Gender Distribution - Pie Chart')

# Adjust layout for the first figure
plt.tight_layout()
plt.savefig('Customer_Demographics_Gender_Distribution.png')

# Age distribution (Histogram and Boxplot)
plt.figure(num='Customer Demographics: Age Distribution Analysis', figsize=(14, 6))

# Subplot 1: Histogram for Age
plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=15, kde=True, color='skyblue')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Subplot 2: Boxplot for Age
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Age'], color='lightgreen')
plt.title('Age Boxplot (Descriptive Analysis)')
plt.ylabel('Age')

# Adjust layout for the second figure
plt.tight_layout()
plt.savefig('Customer_Demographics_Age_Distribution_Analysis.png')

# Annual Income distribution (Histogram and Density Plot)
plt.figure(num='Customer Financials: Annual Income Distribution', figsize=(14, 6))

# Subplot 1: Histogram for Annual Income
plt.subplot(1, 2, 1)
sns.histplot(df['Annual Income (k$)'], bins=15, color='salmon')
plt.title('Annual Income Distribution (Histogram)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')

# Subplot 2: Density Plot for Annual Income
plt.subplot(1, 2, 2)
sns.kdeplot(df['Annual Income (k$)'], color='salmon', fill=True)
plt.title('Annual Income Distribution (Density Plot)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Density')

# Adjust layout and display all exploratory plots
plt.tight_layout()
plt.savefig('Customer_Financials_Annual_Income_Distribution.png')
print("\n[NOTE] Please close the open visualization windows to proceed to K-Means clustering...")
plt.show()

# ==========================================
# Determining Optimal Clusters
# ==========================================
print("\n--- Determining Optimal Clusters ---")
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 1. Elbow Method
iss = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=123)
    kmeans.fit(X)
    iss.append(kmeans.inertia_)

plt.figure(num='Cluster Optimization: Elbow Method Evaluation', figsize=(8, 5))
plt.plot(k_values, iss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters K')
plt.ylabel('Total intra-clusters sum of squares (WCSS)')
plt.grid(True)
plt.savefig('Cluster_Optimization_Elbow_Method_Evaluation.png')

# 2. Average Silhouette Method
silhouette_scores = []
k_values_sil = range(2, 11)
for k in k_values_sil:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=123)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(num='Cluster Optimization: Average Silhouette Score Analysis', figsize=(8, 5))
plt.plot(k_values_sil, silhouette_scores, marker='s', linestyle='-', color='g')
plt.title('Silhouette Method For Optimal k')
plt.xlabel('Number of clusters K')
plt.ylabel('Average Silhouette Score')
plt.grid(True)
plt.savefig('Cluster_Optimization_Average_Silhouette_Score_Analysis.png')

# ==========================================
# Executing K-Means with Optimal k=6
# ==========================================
print("\n--- Executing K-Means with Optimal k=6 ---")
kmeans_opt = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=50, random_state=123)
cluster_labels = kmeans_opt.fit_predict(X)

# Formatting cluster names
df['Cluster'] = ['Cluster ' + str(c + 1) for c in cluster_labels]
cluster_order = sorted(df['Cluster'].unique())

print("Cluster Centers:\n", kmeans_opt.cluster_centers_)

# ==========================================
# Visualizing the Clustering Results
# ==========================================

# Visualization 1: Annual Income vs Spending Score
plt.figure(num='Customer Segmentation: Income vs Spending Score Clusters', figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10', hue_order=cluster_order, s=100, alpha=0.8)
plt.title('Segments of Mall Customers\n(Annual Income vs Spending Score)', fontsize=14)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('Customer_Segmentation_Income_vs_Spending_Score_Clusters.png')

# Visualization 2: Age vs Spending Score
plt.figure(num='Customer Segmentation: Age vs Spending Score Clusters', figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='tab10', hue_order=cluster_order, s=100, alpha=0.8)
plt.title('Segments of Mall Customers\n(Age vs Spending Score)', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('Customer_Segmentation_Age_vs_Spending_Score_Clusters.png')

# Visualization 3: Using the First Two Principal Components (PCA)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
df['PCA1'] = principal_components[:, 0]
df['PCA2'] = principal_components[:, 1]

plt.figure(num='Customer Segmentation: 2D PCA Dimensionality Reduction', figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', hue_order=cluster_order, s=100, alpha=0.8)
plt.title('Segments of Mall Customers\n(Using First Two Principal Components)', fontsize=14)
plt.xlabel('Principal Component 1 (PCA1)')
plt.ylabel('Principal Component 2 (PCA2)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('Customer_Segmentation_2D_PCA_Dimensionality_Reduction.png')

# Save the K-Means Model for future use
joblib.dump(kmeans_opt, 'kmeans_model.pkl')
print("\n[SUCCESS] Saved plots and model ('kmeans_model.pkl').")
print("[SUCCESS] Displaying K-Means cluster visualizations. Close the plots to exit.")
plt.show()
