# K Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# using elbow method
from sklearn.cluster import KMeans
WCSS = []
for i in range (1,11):
    kmeans = KMeans(i, init="k-means++", max_iter=300, n_init = 10)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

# visualizign elbow method
plt.plot(range(1,11), WCSS)
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("WCSS") 
plt.grid()   
plt.show()

# applying kmeans++
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# visualizing clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'gold', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()