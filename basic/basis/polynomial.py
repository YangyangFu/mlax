import jax.numpy as jnp

class Polynomial():
  
    def transform(self, x, degree=3):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        return jnp.array([x**i for i in range(degree + 1)]).T