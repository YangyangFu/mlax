from .base_regression import BaseRegression
import jax.numpy as jnp 

class RidgeRegression(BaseRegression):

    def fit(self, X, y, alpha=1.0):
        """
        Least squares solution for linear regression
        
        Reference: Eq. (3.28) in Pattern Recognition and Machine Learning by Bishop
        
        Note this solution is efficient for small datasets. For large datasets, use sequential learning algorithms to deal with batch data.
        """
        
        eye = jnp.eye(jnp.size(X, 1))
        w = jnp.linalg.solve(
            alpha * eye + X.T @ X,
            X.T @ y,
        )
        
        var = jnp.var(y - X @ w)
        return w, var
    
    def predict(self, w, var, X):
        y = X @ w
        y_std = jnp.sqrt(var) + jnp.zeros_like(y)
        
        return y, y_std