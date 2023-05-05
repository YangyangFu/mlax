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
        
        # Get the indices of the k nearest neighbors for each test point
        # The indices correspond to the row indices in the training data
        # we can use a heap to get the k smallest distances
        nearest_neighbor_indices = jnp.argpartition(distances, k, axis=1)[:, :k]

        # Get the labels of the k nearest neighbors for each test point
        nearest_neighbor_labels = jnp.take_along_axis(y_train.reshape(-1,1), nearest_neighbor_indices, axis=0)

        # Define a JAX-jit function to calculate the mode (most common label) for each test point
        def get_mode(labels):
            # Get the unique labels and their counts
            unique_labels, counts_per_label = jnp.unique(labels, return_counts=True, size=labels.size)
            # Get the probability of each label
            prob = counts_per_label / k
            # Return the most common label
            return unique_labels[jnp.argmax(counts_per_label)], unique_labels, prob

        # Predict the label for each test point by taking the mode of the labels of its k nearest neighbors
        y_pred, ys_pred, prob  = jnp.apply_along_axis(get_mode, axis=1, arr=nearest_neighbor_labels)

        # return predicted label, all possible labels, and the probability of each label
        return y_pred, ys_pred, prob
        

    