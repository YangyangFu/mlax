from jax import vmap
import jax.numpy as jnp

class Linear():
    def _kernel(self, params, x, x_prime):
        """_summary_

        Args:
            x (jnp.array, (D,)): input vector
            x_prime (jnp.array, (D,)): input vector

        Returns:
            float: scalar output of linear kernel on two vectors
        """
        del params 

        return jnp.dot(x,x_prime)
    
    def kernel(self, params, x, x_prime):
        return vmap(vmap(self._kernel, in_axes=(None, None, 0)),
                    in_axes=(None, 0, None))(params, x, x_prime)
 
class Polynomial():
    def _kernel(self, params, x, x_prime):
        """_summary_

        Args:
            x (jnp.array, (D,)): input vector
            x_prime (jnp.array, (D,)): input vector
            degree (int, optional): polynomial degree. Defaults to 3.
            constant (float, optional): constant for polynomial. Defaults to 0..

        Returns:
            float: scalar output of polynomial kernel on two vectors
        """
        
        return (jnp.dot(x,x_prime) + params['constant'])**params['degree']
        
    def kernel(self, params, X, X_prime):
        """
        get kernelmatrix for polynomial kernel
        
        k(X,Y) = (X@Y^T + c)^M
        
        x: (N, D)
        y: (M, D)
        
        
        """
        return vmap(vmap(self._kernel, in_axes=(None, None, 0)), 
                          in_axes=(None, 0, None))(params, X, X_prime)


class RadialBasisFunction():
    def _kernel(self, params, x, x_prime):
        """_summary_

        Args:
            x (jnp.array, (D,)): input vector
            x_prime (jnp.array, (D,)): input vector
            gamma (float, optional): gamma for rbf kernel. Defaults to 1..

        Returns:
            float: scalar output of rbf kernel on two vectors
        """
        
        return params['variance'] * jnp.exp(-0.5 * params['length_scale'] * jnp.sum((x - x_prime)**2))
        
    def kernel(self, params, X, X_prime):
        """
        get kernelmatrix for rbf kernel
        
        k(X,Y) = exp(-gamma * ||X-Y||^2)
        
        x: (N, D)
        y: (M, D)
        
        
        """
        return vmap(vmap(self._kernel, in_axes=(None, None, 0)), 
                          in_axes=(None, 0, None))(params, X, X_prime)