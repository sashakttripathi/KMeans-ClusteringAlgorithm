import random

import numpy as np
from numpy.random import uniform
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# this will generate 5 gaussian distribution clusters and return their coordinates
# and cluster labels
x_sample, y_sample = make_blobs(n_samples=500, centers=5, random_state=42)


# the distance of a dataset from a single point which is going to be the proposed
# centroid. this is a matrix operation we subtract centroid from each row of data
# square it and create a same shape[0] matrix as data, then we square-root each row
def distance_calculator(centroid, data):
    return np.sqrt(np.sum((centroid - data) ** 2, axis=1))


def select_centroid_random(x_training, num_of_clusters):
    data_least_point, data_biggest_point = np.min(x_training, axis=0), np.max(x_training, axis=0)
    return [uniform(data_least_point, data_biggest_point) for _ in range(num_of_clusters)]


def select_centroid_kpp(x_training, num_of_clusters):
    centroids = list()
    centroids.append(random.choice(x_training))
    for _ in range(num_of_clusters - 1):
        distances = np.sum([distance_calculator(centroid, x_training) for centroid in centroids], axis=0)
        distances /= np.sum(distances)
        new_centroid_id = np.random.choice(range(len(x_training)), size=1, p=distances)
        centroids.append(x_training[new_centroid_id][0])
    # print(centroids)
    return centroids


class KMeans:
    def __init__(self, num_of_clusters=5, max_iterations=500):
        self.num_of_clusters = num_of_clusters
        self.max_iterations = max_iterations
        self.centroids = None

    def fit_model(self, x_training):
        # self.centroids = select_centroid_random(x_training, self.num_of_clusters)
        self.centroids = select_centroid_kpp(x_training, self.num_of_clusters)
        # iteration variables
        previous_centroids = []
        # print(self.centroids)
        for i in range(self.max_iterations):
            centroid_points_map = [[] for _ in range(self.num_of_clusters)]
            for x in x_training:
                distance = distance_calculator(x, self.centroids)
                centroid_assigned_index = np.argmin(distance)
                centroid_points_map[centroid_assigned_index].append(x)

            previous_centroids = self.centroids
            self.centroids = [np.mean(centroid_cluster, axis=0) for centroid_cluster in centroid_points_map]

            for j, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[j] = previous_centroids[j]

            if np.equal(previous_centroids, self.centroids).all():
                break

    def evaluate(self, x):
        centroids = []
        centroid_indexes = []
        for item in x:
            distances = distance_calculator(item, self.centroids)
            centroid_index = np.argmin(distances)
            centroids.append(self.centroids[centroid_index])
            centroid_indexes.append(centroid_index)

        return centroids, centroid_indexes


kmeans = KMeans()
kmeans.fit_model(x_training=x_sample)

class_centers, classification = kmeans.evaluate(x_sample)
sns.scatterplot(x=[x[0] for x in x_sample], y=[x[1] for x in x_sample], hue=y_sample, palette='deep', legend=None)
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         '+', markersize=10)
plt.show()
