import numpy as np

class KMeans():
    def __init__(self):
        self.centroids = []
        self.distance_function = None
        self.max_iterations = 10000
    
    def initialize_random(self, centroid_number, data):
        self.centroids = np.random.choice(data, centroid_number)
        return self
        
    def initialize_plus_plus(self, centroid_number, data):
        self.centroids = np.random.choice(data, centroid_number)
        return self

    def with_euclidian_distance(self):
        self.distance_function = np.linalg.norm
        return self
    
    def set_max_iterations(self, iterations):
        self.max_iterations = iterations

    def train(self, X, learning_rate):
        current_iteration = 0
        while(current_iteration < self.max_iterations):
            self._fit(X, learning_rate)
        return self
    
    def predict(self, x):
        pass

    def _get_bmu(self, x):
        bmu_id, bmu, bmu_dist = 0, self.centroids[0], self.distance_function(self.centroids[0] - x)
        for i, c in enumerate(self.centroids[1:]):
            c_dist = self.distance_function(c - x)
            if c_dist < bmu_dist:
                bmu_id, bmu, bmu_dist = i, c, c_dist
        return bmu_id, bmu


    def _fit(self, X, learning_rate):
        groups = [[] for i in self.centroids]
        for x in X:
            bmu_id, _ = self._get_bmu(x)
            groups[bmu_id].append(x)
        
        for group_id in enumerate(groups):
            group_mean = np.mean(groups[group_id], 0)
            self.centroids[group_id] = group_mean
