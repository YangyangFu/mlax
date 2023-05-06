import jax
import jax.numpy as jnp
import jax.random as random
from basic.utils.distance import cdist

class KMeans():
    
    def fit(self, rng_key, k, X, max_iter=100):
        """Perform k-means clustering.
        
        """
        n_samples, n_features = X.shape
        I = jnp.eye(k)

        # check X data type in case of type casting error for images (unit8)
        X = X.astype(jnp.float32)

        # while loop in jax to support jit
        # Define the condition function
        def cond_fun(state):
            iter_count, _, _, _ = state
            return iter_count < max_iter

        # Define the body function
        def body_fun(state):
            iter_count, centers, prev_centers, _ = state

            # Calculate distances between points and centroids
            D = cdist(X, centers)

            # Assign points to the closest centroid
            cluster_index = jnp.argmin(D, axis=1)

            # one-hot encoding: each row is a point, each column is a cluster index as one-hot vector
            index = I[cluster_index]
            
            # update centroids using the mean of the points in each cluster
            # need manipulate the shape of X and cluster_index to make sure the dimensions match for broadcasting
            # X: (n_samples, n_features) -> (n_samples, 1, n_features)
            # cluster_index: (n_samples, n_clusters) -> (n_samples, n_clusters, 1)
            centers = jnp.sum(X[:, None, :] * index[:, :, None], axis=0) / jnp.sum(index, axis=0)[:, None]

            # Check for convergence
            def true_fcn(_):
                return max_iter 
            
            def false_fcn(_):
                return iter_count + 1
            
            iter_count = jax.lax.cond(jnp.allclose(prev_centers, centers), true_fcn, false_fcn, None)
            
            # the following code is discarded due to error in jit
            #if jnp.allclose(prev_centers, centers):
            #    iter_count = max_iter  # Set iter_count to max_iter to exit the loop
            #else:
            #    iter_count += 1

            return iter_count, centers, jnp.copy(centers), cluster_index
        
        # Initial state
        iter_count_init = 0
        centers = random.choice(rng_key, X, shape=(k,), replace=False)
        prev_centers_init = jnp.copy(centers)
        cluster_index_init = jnp.zeros(n_samples, dtype=jnp.int32)  # Initial cluster_index with zeros
        state_init = (iter_count_init, centers, prev_centers_init, cluster_index_init)

        # Run the while loop
        _, centers_final, _, cluster_index = jax.lax.while_loop(cond_fun, body_fun, state_init)

        return centers_final, cluster_index
        
    
    def predict(self, X, centers, cluster_index):
        """Predict cluster index for each sample.
        
        """
        
        D = cdist(X, centers)
        cluster_index = jnp.argmin(D, axis=1)
              
        return cluster_index