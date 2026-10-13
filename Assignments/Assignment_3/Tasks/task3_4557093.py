# No external libraries are allowed to be imported in this file
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to generate the dataset
def generate_swiss_roll(n_samples, noise=0.1, random_state=2024):
    """
    Generates the Swiss Roll dataset.

    Parameters:
    n_samples (int): Number of samples to generate.
    noise (float): Noise factor.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Generated data (X) and the color labels (color)
    """
    X, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, color

# Function to apply PCA to the dataset
def apply_pca(X, n_components, random_state=2024):
    """
    Applies PCA to the Swiss Roll dataset.

    Parameters:
    X (array): The input dataset.
    n_components (int): Number of principal components to retain.
    random_state (int): Random seed for reproducibility.

    Returns:
    array: Transformed data with PCA applied.
    """
    # TO DO: Create a pipeline to apply StandardScaler and PCA
    pipe = make_pipeline(StandardScaler(), PCA(n_components=n_components,random_state=random_state))
    X_pca = pipe.fit_transform(X)

    return X_pca

# Function to plot the original 3D data
def plot_3d_data(X, color):
    """
    Plots the 3D Swiss Roll dataset.

    Parameters:
    X (array): The 3D dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the original data in 3D space
    fig, ax = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    ax = ax.flatten()
    ax[0].scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=color)
    ax[0].set_title("Original Dataset")
    ax[0].set_title("Default View")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_zlabel("Z")
    ax[1].scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=color)
    ax[1].view_init(elev=90, azim=-90)
    ax[1].set_title("XY")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].get_zaxis().set_ticks([])
    ax[2].scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=color)
    ax[2].view_init(elev=0, azim=-90)
    ax[2].set_title("XZ")
    ax[2].set_xlabel("X")
    ax[2].set_zlabel("Z")
    ax[2].get_yaxis().set_ticks([])
    ax[3].scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=color)
    ax[3].view_init(elev=0, azim=0)
    ax[3].set_title("YZ")
    ax[3].set_ylabel("Y")
    ax[3].set_zlabel("Z")
    ax[3].get_xaxis().set_ticks([])
    plt.suptitle("3D View")
    plt.tight_layout()
    plt.show()




# Function to plot the XZ projection
def plot_xz_projection(X, color):
    """
    Plots the XZ projection of the Swiss Roll dataset.

    Parameters:
    X (array): The 3D dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the XZ projection

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=color)
    ax.view_init(elev=0, azim=-90)
    ax.set_title("XZ")
    ax.set_xlabel("X")
    ax.set_zlabel("Z")
    ax.get_yaxis().set_ticks([])
    plt.show()



# Function to plot the 2D PCA projection
def plot_pca_projection(X_pca, color):
    """
    Plots the 2D PCA projection of the Swiss Roll dataset.

    Parameters:
    X_pca (array): The PCA-transformed dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the 2D projection from PCA
    plt.scatter(X_pca[:,0],X_pca[:,1],c=color)
    plt.title("PCA Projection")
    #I don't know if it's correct to call the axis X and Y because its a reprojection
    plt.xlabel("PCA_1")
    plt.ylabel("PCA_2")
    plt.show()




if __name__ == "__main__":
    np.random.seed(2024)
    X, color = generate_swiss_roll(n_samples=1500, noise=0.1, random_state=2024)

    #TO DO: Fill in appropriate value for n_components
    X_pca = apply_pca(X, n_components=2, random_state=2024) # Apply PCA

    plot_3d_data(X, color)              # Visualize the original 3D dataset
    plot_xz_projection(X, color)        # Visualize the XZ projection
    plot_pca_projection(X_pca, color)   # Visualize the PCA 2D projection

    # print('Observation is that for the 3D projection there is a swiss roll looking object that curls inwards clockwise along one axis')
    # print('Observation is that for the 2D projection is a flat field where groups overlap but have general areas associated with their color, It looks like the XY projection')
    # print('Hey, I don\'t work here, but maybe I should')
