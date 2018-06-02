import numpy as np
from k_means import KMeans

    # .initialize_random()
if __name__ == '__main__':
    kmeans = KMeans() \
                    .initialize_random(3,) \
                    .with_euclidian_distance() \
                    .set_max_iterations(10000)