import jax.numpy as jnp 
import jax.scipy as jsp

class GaussianProcessRegression():

    def compute_posterior(self, kernel, params, X_train, y_train, X_test, noise_variance):
        # compute convariance matrices C_{n+1} = [C_n, k; k.T, c_{n+1}]
        K = kernel.kernel(params, X_train, X_train)
        K_s = kernel.kernel(params, X_test, X_test)
        K_sT = kernel.kernel(params, X_test, X_train)
        
        # directly computing the inverse is numerically unstable
        # K_inv = jnp.linalg.inv(K + noise_variance * jnp.eye(K.shape[0]))
        # we use scipy.linalg.solve instead
        # K*K_inv = I
        # refer to Eq.(6.62) in PRML
        K_inv = jsp.linalg.solve(K + noise_variance * jnp.eye(len(X_train)), jnp.eye(len(X_train)))

        # posterior mean
        mu_s = jnp.matmul(jnp.matmul(K_sT, K_inv), y_train)
        
        # posteiror covariance
        cov_s = K_s - jnp.matmul(jnp.matmul(K_sT, K_inv), K_sT.T)
        
        # diagonal elements of the covariance matrix
        #var_s = jnp.diag(cov_s) 
        
        # return 
        return mu_s, cov_s
        
    # Make predictions
    def predict(self, kernel, params, X_train, y_train, X_test, noise_variance):
        mu_s, cov_s = self.compute_posterior(kernel, params, X_train, y_train, X_test, noise_variance)
        return mu_s, cov_s
        