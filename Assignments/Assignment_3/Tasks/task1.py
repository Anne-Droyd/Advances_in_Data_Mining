# No external libraries are allowed to be imported in this file
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Function to generate the dataset
def generate_blobs_dataset(n_samples, centers, n_features, random_state):
    """
    Generates a 3D dataset using make_blobs.

    Parameters:
    n_samples (int): Number of samples.
    centers (int): Number of centers.
    n_features (int): Number of features (3 for 3D data).
    random_state (int): Seed for reproducibility.

    Returns:
    tuple: Generated data (X, y)
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=random_state)
    return X, y

# Function to plot the original 3D data
def plot_3d_data(X, y):
    # TO DO: Use scatter plot to visualize the original data in 3D space
    #X is 3d vector, y is binary
    #y is actually a cluster label so rn there is only 2 labels so it is binary, but it could range until inf
    # I don't technically need to show the cluster but I will in colors

    fig, ax = plt.subplots(2,2,subplot_kw={'projection':'3d'})
    ax = ax.flatten()
    ax[0].scatter(xs = X[:,0],ys = X[:,1],zs = X[:,2], c=y)
    ax[0].set_title("Original Dataset")
    ax[0].set_title("Default View")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_zlabel("Z")
    ax[1].scatter(xs = X[:,0],ys = X[:,1],zs = X[:,2],c=y)
    ax[1].view_init(elev=90, azim=-90)
    ax[1].set_title("XY")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].get_zaxis().set_ticks([])
    ax[2].scatter(xs = X[:,0],ys = X[:,1],zs = X[:,2],c=y)
    ax[2].view_init(elev=0, azim=-90)
    ax[2].set_title("XZ")
    ax[2].set_xlabel("X")
    ax[2].set_zlabel("Z")
    ax[2].get_yaxis().set_ticks([])
    ax[3].scatter(xs = X[:,0],ys = X[:,1],zs = X[:,2],c=y)
    ax[3].view_init(elev=0, azim=0)
    ax[3].set_title("YZ")
    ax[3].set_ylabel("Y")
    ax[3].set_zlabel("Z")
    ax[3].get_xaxis().set_ticks([])
    plt.suptitle("3D View")
    plt.tight_layout()
    plt.show()

    #seems too difficult to label the classes here so I'll pass

# Function to standardize the dataset
def standardize_data(X):
    """
    Standardizes the dataset using StandardScaler.

    Parameters:
    X (array): Original dataset.

    Returns:
    array: Scaled dataset.
    """
    # TO DO: Instantiate StandardScaler and use fit_transform to scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

# Function to apply PCA to the dataset
def apply_pca(X_scaled, n_components):
    """
    Applies PCA to reduce dimensionality of the dataset.

    Parameters:
    X_scaled (array): Scaled dataset.
    n_components (int): Number of principal components.

    Returns:
    array: PCA transformed data.
    PCA object: The PCA object used.
    """
    # TO DO: Instantiate PCA with n_components and apply PCA transformation

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca

# Function to plot the 2D PCA projection
def plot_pca_projection(X_pca, y):
    # TO DO: Use scatter plot to visualize the 2D projection from PCA
    plt.scatter(X_pca[:,0],X_pca[:,1],c=y)
    plt.title("PCA Projection")
    #I don't know if it's correct to call the axis X and Y because its a reprojection
    plt.xlabel("PCA_1")
    plt.ylabel("PCA_2")
    plt.show()


if __name__ == "__main__":
    np.random.seed(2024)
    X, y = generate_blobs_dataset(n_samples=300, centers=2, n_features=3, random_state=2024)
    plot_3d_data(X, y)  # Visualize the original 3D dataset
    X_scaled = standardize_data(X)  # Standardize the dataset

    #TO DO: Fill in appropriate value for n_components
    # is this a trick, its reduction from ndim 3 to ndim 2?
    X_pca, pca = apply_pca(X_scaled, n_components=2)  # Apply PCA

    plot_pca_projection(X_pca, y)  # Visualize the 2D PCA projection

    print('Observation is that for the 3D projection they are two distinct groups')
    print('Observation is that for the 2D projection they are two distinct groups but at a slope')

