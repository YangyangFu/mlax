import jax.numpy as jnp
import jax.random as random
from scipy.spatial.distance import cdist

class KMeans():
    
    def fit(self, rng_key, k, X, max_iter=100):
        """Perform k-means clustering.
        
        """
        
        I = jnp.eye(k)
        # initialize centroids
        centers = random.choice(rng_key, X, shape=(k,), replace=False)
        
        # this needs to be modified to use scan for jax.jit
        # iterate until convergence
        for _ in range(max_iter):
            prev_centers = jnp.copy(centers)
            # calculate distances between points and centroids
            # TODO: this is a non-JAX library, which is not jittable- need reimplement the distance function using jnp
            D = cdist(X, centers, metric='euclidean')
            # assign points to the closest centroid
            cluster_index = jnp.argmin(D, axis=1)
            # one-hot encoding: each row is a point, each column is a cluster index as one-hot vector
            index = I[cluster_index]
            
            # update centroids using the mean of the points in each cluster
            # need manipulate the shape of X and cluster_index to make sure the dimensions match for broadcasting
            # X: (n_samples, n_features) -> (n_samples, 1, n_features)
            # cluster_index: (n_samples, n_clusters) -> (n_samples, n_clusters, 1)
            centers = jnp.sum(X[:, None, :] * index[:, :, None], axis=0) / jnp.sum(index, axis=0)[:, None]
            
            if jnp.allclose(prev_centers, centers):
                break
                
        return centers, cluster_index
        
    
    def predict(self, X, centers, cluster_index):
        """Predict cluster index for each sample.
        
        """
        
        D = cdist(X, centers, metric='euclidean')
        cluster_index = jnp.argmin(D, axis=1)
              
        return cluster_index