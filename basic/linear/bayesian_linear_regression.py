from .base_regression import BaseRegression
import jax.numpy as jnp

class BayesianLinearRegression(BaseRegression):
    
    def init_w_prior(self, X, alpha = 1):
        """
        Initialize parameter w prior distribution
        
        Default: zero mean and identity covariance matrix
        
        X: input data, shape (N, D)
        alpha: precision of the prior distribution of w
        """
        
        return jnp.zeros(X.shape[1]), (1./alpha)*jnp.eye(X.shape[1])
    
    def fit(self, beta, w_mean_prior, w_var_prior, X, y):
        """
        Bayesian linear regression
        
        Reference:
            Eq. (3.49-3.51) in Pattern Recognition and Machine Learning by Bishop
        
        beta: precision of the noise distribution
        w_mean_prior: mean of the prior distribution of w
        w_var_prior: variance of the prior distribution of w
        
        """
        # w = (X.T @ X + alpha * I)^-1 @ X^T @ y
        w_precision_prior = jnp.linalg.inv(w_var_prior)
        w_precision = w_precision_prior + beta * X.T @ X
        w_mean = jnp.linalg.inv(w_precision) @ (w_precision_prior @ w_mean_prior + beta * X.T @ y)
        w_var = jnp.linalg.inv(w_precision)
        
        return w_mean, w_var 

    def predict(self, beta, w_mean, w_var, X):
        
        """Predictive distribution

        Reference:
            Eq. (3.58) in Pattern Recognition and Machine Learning by Bishop
        """
        y = w_mean.T @ X
        
        y_var = 1./beta + X.T @ w_var @ X
        y_std = jnp.sqrt(y_var)
        
        return y, y_std