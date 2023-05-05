import jax.numpy as jnp 
from scipy.spatial.distance import cdist

class NearestNeighborsDensity():
    def predict(self, X, x, k):
        """Estimate the density of each sample.
        
        """
        
                # Calculate the pairwise distances between test points and training points
        # using the SciPy cdist function (Euclidean distance by default)
        distances = cdist(x, X)
        
        pass