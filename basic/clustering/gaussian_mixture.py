import jax 
import jax.numpy as jnp 
from jax.scipy.stats import multivariate_normal as norm
from .k_means import KMeans

class GaussianMixture():
    def __init__(self, num_components):
        self.num_components = num_components 
    
    def initialize_parameters(self, rng_key, data):
        """ Use kmeans to initialize the parameters of the model 
        """
        # instantiate a kmeans object
        kmeans = KMeans()
        
        # kmeans gives the k centroids
        means, clusters = kmeans.fit(rng_key, self.num_components, data)
        
        # compute the covariance for each cluster
        cov = jnp.stack([jnp.cov(data[clusters == i].T) for i in range(self.num_components)])
        
        # initialize the mixing probabilities
        mixing_probs = jnp.ones(self.num_components) / self.num_components

        return means, cov, mixing_probs
    
    def get_responsibilities(self, data, means, cov, mixing_probs):
        
        # Discard to support broadcasting
        #for k in range(self.num_components):
            
        #    responsibilities = responsibilities.at[:,k].set(mixing_probs[k] * norm.pdf(data, means[k,:], cov[k,:,:]))
        # (N, K) = (K,) * (N, K)
        responsibilities = mixing_probs * norm.pdf(data[:, None], means, cov)        
        responsibilities /= jnp.sum(responsibilities, axis=1, keepdims=True)

        return responsibilities
    
    # Define the E step
    def e_step(self, data, means, cov, mixing_probs):
        return self.get_responsibilities(data, means, cov, mixing_probs)

    # Define the M step
    def m_step(self, data, responsibilities):
        
        mixing_probs = jnp.mean(responsibilities, axis=0)
        
        # discard the following to support broadcasting
        #means = jnp.zeros((self.num_components, data.shape[1]))
        #cov = jnp.zeros((self.num_components, data.shape[1], data.shape[1]))
        
        #for k in range(self.num_components):
        #    means_k = jnp.sum(responsibilities[:,k] * data.T, axis=1) / (mixing_probs[k] * data.shape[0])
        #    means = means.at[k,:].set(means_k)
        #    x = (data - means_k) 
        #    # x is (N,D), and xx is a (N, D, D) matrix
        #    xx = (x[:, :, None] * x[:, None, :])
        #    # cov_k is a (D, D) matrix
        #    cov_k = jnp.sum(responsibilities[:,k][:,None, None] * xx, axis=0) / (mixing_probs[k] * data.shape[0])
        #    cov = cov.at[k].set(cov_k)
        
        # Broadcasting
        # (K, D) = sum((N, K, 1) * (N, 1, D))
        means = jnp.sum(responsibilities[:, :, None] * data[:, None, :], axis=0) / (mixing_probs[:, None] * data.shape[0])

        # (N, K, D) = (N, 1, D) - (1, K, D)
        X = data[:, None, :] - means[None, :, :]
        # (N, K, D, D) = (N, K, D, 1) * (N, K, 1, D)
        XX = X[:, :, :, None] * X[:, :, None, :]
        # (K, D, D) = sum((N, K, 1, 1) * (N, K, D, D))
        cov = jnp.sum(responsibilities[:, :, None, None]*XX, axis=0) / (mixing_probs[:, None, None] * data.shape[0])

        return means, cov, mixing_probs

    def log_likelihood(self, data, means, cov, mixing_probs):
        # (N, K) = (K,) * (N, K)
        responsibilities = mixing_probs * norm.pdf(data[:, None], means, cov) 
        
        # (N, 1)
        log_likelihood = jnp.sum(jnp.log(jnp.sum(responsibilities, axis=1)))
        
        return log_likelihood

    # Define fit using E and M steps
    def fit(self, data, rng_key, num_steps=100):
        
        # iteration parameters
        step = 0
        rel_log_likelihood = 1
        rel_tol = 1e-5
        
        # initialize the parameters
        means_prev, cov_prev, mixing_probs_prev = self.initialize_parameters(rng_key, data)
        # initialize the log likelihood
        log_likelihood_prev = self.log_likelihood(data, means_prev, cov_prev, mixing_probs_prev)

        # main loop
        while step < num_steps and rel_log_likelihood > rel_tol:
            # e_step
            responsibilities = self.e_step(data, means_prev, cov_prev, mixing_probs_prev)
            # m_step
            means, cov, mixing_probs = self.m_step(data, responsibilities)
            
            # calculate the log likelihood
            log_likelihood = self.log_likelihood(data, means, cov, mixing_probs)
            rel_log_likelihood = jnp.abs(log_likelihood - log_likelihood_prev) / (jnp.abs(log_likelihood_prev) + 1e-6)
            print("At step: {}, Log likelihood: {}, Relative log likelihood: {}".format(step, log_likelihood, rel_log_likelihood))
            
            # stop if parameter converge 
            if jnp.allclose(means, means_prev) and jnp.allclose(cov, cov_prev) and jnp.allclose(mixing_probs, mixing_probs_prev):
                print("Parameters converged at step {}".format(step))
                print("Log likelihood: {}, Relative log likelihood: {}".format(log_likelihood, rel_log_likelihood))
                break
            
            # update state
            step += 1
            means_prev = means.copy()
            cov_prev = cov.copy()
            mixing_probs_prev = mixing_probs.copy()
        
        return means, cov, mixing_probs, responsibilities
    
    # define a sampling function
    def sampling(self, rng_key, n_samples, means, cov, mixing_probs):
        """ Sample from the GMM based on ancestral sampling
            
        """
        # sample from the mixing probabilities
        z = jax.random.choice(rng_key, self.num_components, shape=(n_samples,), p=mixing_probs)
        # sample from the corresponding multivariate normal
        x = jax.random.multivariate_normal(rng_key, means[z], cov[z])
        return x