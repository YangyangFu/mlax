import jax.numpy as jnp 
import jax.random as random
from scipy.spatial.distance import cdist
import jax 

class NearestNeighborsClassifier:
    def predict(self, X_train,y_train, X_test, k):
        """Make a prediction for each sample

            - calculate the distance between the query point and 
                all the data points in the training dataset.
            - sort the distances in ascending order.
            - determine the class label for the query points by 
                taking a majority vote among the class labels of 
                the k nearest neighbors.
            
            Returns:
                y_pred: predicted class labels for each sample in X_test
        """
        
        
        # Calculate the pairwise distances between test points and training points
        # using the SciPy cdist function (Euclidean distance by default)
        distances = cdist(X_test, X_train)
        print(distances.shape)
        # Get the indices of the k nearest neighbors for each test point
        # The indices correspond to the row indices in the training data
        # we can use a heap to get the k smallest distances
        nearest_neighbor_indices = jnp.argpartition(distances, k, axis=1)[:, :k]
        print(distances)
        print(nearest_neighbor_indices)
        print(X_train.shape, y_train.shape, X_test.shape, k, nearest_neighbor_indices.shape)
        # Get the labels of the k nearest neighbors for each test point
        nearest_neighbor_labels = jnp.take_along_axis(y_train.reshape(-1,1), nearest_neighbor_indices, axis=0)

        # Define a JAX-jit function to calculate the mode (most common label) for each test point
        def get_mode(labels):
            unique_labels, counts = jnp.unique(labels, return_counts=True, size=labels.size)
            return unique_labels[jnp.argmax(counts)]

        # Predict the label for each test point by taking the mode of the labels of its k nearest neighbors
        test_y = jnp.apply_along_axis(get_mode, axis=1, arr=nearest_neighbor_labels)

        return test_y
        

    