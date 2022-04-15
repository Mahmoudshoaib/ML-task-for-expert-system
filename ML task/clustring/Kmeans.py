import pandas as pd  # data processing
import matplotlib.pyplot as plt  # Data Visualization

# Import the dataset

dataset = pd.read_csv('Mall_Customers.csv')

# Exploratory Data Analysis
# As this is unsupervised learning so Label (Output Column) is unknown

dataset.head(10)  # Printing first 10 rows of the dataset

dataset.info()  # there are no missing values as all the columns has 200 entries properly

### Feature selection for the model

X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans

wcss = []


### Static code to get max no of clusters

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    # inertia_ is the formula used to segregate the data points into clusters

# # Visualizing the ELBOW method to get the optimal value of K
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('no of clusters')
# plt.ylabel('wcss')
# plt.show()

# Model Build
kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)

# Visualizing all the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

###Model Interpretation
# Cluster 1 (Red Color) -> earning high but spending less
# cluster 2 (Blue Colr) -> average in terms of earning and spending
# cluster 3 (Green Color) -> earning high and also spending high [TARGET SET]
# cluster 4 (cyan Color) -> earning less but spending more
# Cluster 5 (magenta Color) -> Earning less , spending less
