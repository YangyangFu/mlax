from .base_regression import BaseRegression
import jax.numpy as jnp 

class LinearRegression(BaseRegression):

    def fit(self, X, y):
        """
        Least squares solution for linear regression
        
        Reference: Eq. (3-15) and (3-21) in Pattern Recognition and Machine Learning by Bishop
        
        Note this solution is efficient for small datasets. For large datasets, use sequential learning algorithms to deal with batch data.
        """
        # w = (X.T @ X)^-1 @ X^T @ y
        w = jnp.linalg.pinv(X) @ y
        var = jnp.var(y - X @ w)
        return w, var
    
    def predict(self, w, var, X):
        y = X @ w
        y_std = jnp.sqrt(var) + jnp.zeros_like(y)
        
        return y, y_std