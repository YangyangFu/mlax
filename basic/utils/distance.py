import jax.numpy as jnp 

def cdist(X,Y):
    """
    This is a simplified version of scipy.spatial.distance.cdist.
    
    Compute the squared Euclidean distances between all pairs of points
    in X and Y. The result is a matrix where the (i, j)-th entry is the squared
    Euclidean distance between the i-th point in X and the j-th point in Y.
    """

    squared_distances = jnp.sum(jnp.square(X[:, jnp.newaxis] - Y), axis=-1)
    
    # Take the square root to obtain the Euclidean distances.
    distances = jnp.sqrt(squared_distances)
    
    return distances