import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

from basic.kernel.kernel import Linear

class BinarySVC():
    """ Support Vector Machine Binary Classifier
     
    """
    kernel = Linear()
    kernel_params = None
    C = 2.0
    tolerance = 1e-6

    def fit(self, X, t):
        """fit model to data
        
        Solve the following dual optimization problem:
            $$ \min_{\alpha} \frac{1}{2}(\alpha t)^T K (\alpha t) - (\alpha t)^T t $$ 
            
            subject to:
            
            $$ 0 \leq \alpha_i \leq C $$
            $$ \sum_{i=1}^{N} \alpha_i t_i = 0 $$
            
            reformulation by substituting $ \beta = \alpha t $:
            $$ \min_{\beta} \frac{1}{2} \beta^T K \beta - \beta^T 1 $$
            
        Args:
            X (jnp.array, (N, D)): input data
            t (jnp.array, (N,)): target data
            kernel (function): kernel function
            kernel_params (dict): kernel parameters
            C (float): regularization parameter
            
        Returns:
            dict: dictionary of parameters
        """
        
        def matvec_Q(X, beta):
            # the objective implementation in OSQP is 0.5*x^T * matvec_Q(P,x)
            # this returns Kbeta = X X^T beta
            # because OSQP assume 0.5*x^T * matvec_Q(P,x) in the objective
            # return shape: (N,)
            
            Gram = self.kernel.kernel(self.kernel_params, X, X)
            return Gram @ beta

        def matvec_A(_, beta):
            return beta, jnp.sum(beta)
        
        # l, u must have same shape as matvec_A's output.
        l = -jax.nn.relu(-t * self.C), 0.
        u =  jax.nn.relu( t * self.C), 0.
        
        # formulate and solve quadratic programming problem
        hyper_params = dict(params_obj=(X, -t), params_eq=None, params_ineq=(l, u))
        osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=self.tolerance)
        params, _ = osqp.run(init_params=None, **hyper_params)
        beta = params.primal[0]

        # for support vector indices: if true, then the corresponding sample is a support vector
        is_sv = self.get_support_vectors(beta)
        
        return beta, is_sv

    def get_support_vectors(self, beta):
        # this sucks in JAX because it is not jittable due to boolean indexing
        # beta is signed 
        # beta = 0 means the Langrange multiplier is 0, which means the corresponding sample does not contribute to the sum in the objective function.
        # beta ~= 0 means the samples are support vectors
        
        is_sv = jnp.abs(beta) > self.tolerance

        # have to return True/False array instead of indices for True. The latter is not jittable
        #res = jnp.where(is_sc)
        return is_sv
    
    def _accuracy(self):
        """get accuracy of model:
            if 0 < abs(beta) < C, then epsilon = 0, then the sample is on the margin
            if abs(beta) = C, then the sample can lie inside the margin and can either be correctly classied if epsilon <=1 or misclassified if epsilon > 1
        """
        pass
    
    def predict(self, X_test, X_train, y_train, beta, sv_index):
        
        """solving primal problem gives w and b
            From Eq. (7.29) and (7.37) in Bishop's book:
            $$ w = \sum_{i=1}^{N} \alpha_i t_i x_i = (\beta^T x)^T = x^T \beta $$
            $$ wx = w^Tx^T = \beta^T x x^T = \beta^T K$$
            
        """
        # get wx
        Gram = self.kernel.kernel(self.kernel_params, X_train[sv_index], X_test)
        wx = beta[sv_index].T @ Gram 
        
        # get b
        # get indice of support vectors on the margin: 0 < abs(beta) < C
        M_mask = jnp.abs(beta[sv_index]) < self.C-self.tolerance
        Gram_S = self.kernel.kernel(self.kernel_params, X_train[sv_index], X_train[sv_index]) # (S,S)

        # define some jittable functions
        def set_nonmargin_to_zero(x, M):
            return jnp.where(M, x, 0)
        
        def get_nonzero_mean(x):
            return jnp.mean(x, where = x != 0)
                    
        bv = set_nonmargin_to_zero(y_train[sv_index] - Gram_S @ beta[sv_index], M_mask)
        b = get_nonzero_mean(bv)

        # This version is not jittable, and seems slightly different for the final b
        # b1 
        #Gram_M = kernel.kernel(kernel_params, X_train[sv][M_mask], X_train[sv]) # (M, S)
        #bv1 = y_train[sv][M_mask] - Gram_M @ beta[sv]
        #b1 = jnp.mean(bv1)
        #print(bv1.shape, b1, bv1)
        #print(b)
        # retur signs of wx + b: 1 or -1
        return jnp.sign(wx + b)