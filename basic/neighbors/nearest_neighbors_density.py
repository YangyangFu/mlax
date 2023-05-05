import jax
import jax.numpy as jnp 
from scipy.spatial.distance import cdist

class NearestNeighborsDensity():
    def predict(self, X_train, X_test, k):
        """Estimate the density of each sample.
        
        """
        
        # Calculate the pairwise distances between test points and training points
        # using the SciPy cdist function (Euclidean distance by default)
        distances = cdist(X_test, X_train)
        print(distances.shape)
        
        # get the radius of the k nearest neighbors for each test point
        radius = jnp.sort(distances, axis=1)[:, k-1]
        
        # get the volume of the k nearest neighbors for each test point
        # jax seems no gamma function implemented but gamma distribution is implemented
        volume = jnp.pi**(X_train.shape[1]/2) * radius**X_train.shape[1] / jnp.exp(jax.scipy.special.gammaln(X_train.shape[1]/2 + 1))
        
        return k / (volume * X_train.shape[0])
