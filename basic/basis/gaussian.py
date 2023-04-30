import jax.numpy as jnp

class Gaussian():
    
    def transform(self, x, mean, var):
        """_summary_

        Args:
            x ((sample_size, ndim)): _description_
            mean ((n_features, ndim) or (n_features,)) : _description_
            var (float): _description_

        Returns:
            _type_: _description_
        """
        # dimension check
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        assert isinstance(var, float) or isinstance(var, int)
        
        # dimension check
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert jnp.size(x, 1) == jnp.size(mean, 1)
        
        # gaussian basis
        basis = [jnp.ones(len(x))]
        for m in mean:
            trans = jnp.exp(-0.5 * jnp.sum(jnp.square(x - m), axis=-1) / var)
            basis.append(trans)
            
        return jnp.array(basis).transpose()