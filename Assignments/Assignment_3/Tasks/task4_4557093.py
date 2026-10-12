# No external libraries are allowed to be imported in this file
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data():
    """
    Loads the Swiss Roll dataset and corresponding color labels from files.

    Returns:
    tuple: The data (X) and color labels (color)
    """
    # TO DO: Load dataset from files
    X = np.load('../Data/swiss_roll.npy')
    color = np.load('../Data/color.npy')



    return X, color

# Function to apply t-SNE to the dataset
def apply_tsne(X, n_components, perplexity, max_iter, init, random_state=2024):
    """
    Applies t-SNE to the Swiss Roll dataset after scaling it.

    Parameters:
    X (array): The input dataset.
    perplexity (float): t-SNE perplexity parameter.
    random_state (int): Random seed for reproducibility.

    Returns:
    array: The t-SNE transformed dataset with 2 components.
    """
    # TO DO: Create a pipeline to apply StandardScaler and t-SNE

    pipe = make_pipeline(StandardScaler(),
                         TSNE(n_components=n_components,
                              perplexity=perplexity,
                              random_state=random_state,
                              max_iter=max_iter,
                              init=init))

    X_tsne_2d = pipe.fit_transform(X)


    return X_tsne_2d

# Function to plot the 2D t-SNE projection
def plot_tsne_projection(X_tsne_2d, color):
    """
    Plots the 2D projection of the t-SNE transformed Swiss Roll dataset.

    Parameters:
    X_tsne_2d (array): The t-SNE transformed dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the 2D projection from t-SNE
    plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=color)
    plt.title('t-SNE')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()



# Function to return a recogzinable letter from the plot
def return_identified_letter():
    """
    Returns the letter identified from the t-SNE plot.
    """
    # TO DO: If you succeed in unfolding the dataset with t-SNE, you will see a recognizable letter (between A-Z) in the plot.
    # Identify and return the letter. Example: return 'A'.
    # I think this is right because all values are -1 which is abnormal for the data set
    return 'o'.upper()


if __name__ == "__main__":
    X, color = load_data()

    # TO DO: Fill in the appropriate values for n_components, perplexity, max_iter, and init
    X_tsne_2d = apply_tsne(X, n_components=2, perplexity=80, max_iter=500, init='pca', random_state=2024)

    plot_tsne_projection(X_tsne_2d, color)
    print(return_identified_letter())