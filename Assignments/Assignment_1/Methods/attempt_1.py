# Attempted by
# Matthew Kirwan
# s4557093
# 29/9/26
# 20:11



#============================================
# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
#============================================

# What if I make my own packages?






# 1. COSINE SIMILARITY:
def similarity_matrix(matrix, k=5, axis=0):
    """
    This function should contain the code to compute the cosine similarity
    (according to the formula seen at the lecture) between users (axis=0) or
    items (axis=1) and return a dictionary where each key represents a user
    (or item) and the value is a list of the top k most similar users or items,
    along with their similarity scores.

    Args:
        matrix (pd.DataFrame) : user-item rating matrix (df)
        k (int): number of top k similarity rankings to return for each \
                    entity (default=5)
        axis (int): 0: calculate similarity scores between users \
                        (rows of the matrix),
                    1: calculate similarity scores between items <- spelling mistake \
                        (columns of the matrix)

    Returns:
        similarity_dict (dictionary): dictionary where the keys are users
        (or items) and the values are lists of tuples containing the most
        similar users (or items) along with their similarity scores.

    Note that is NOT allowed to automatically compute cosine similarity using <- spelling mistake
    an apposite function from any package, the computation should follow the
    formula that has been discussed during the lecture and that can be found in
    the slides.

    Note that is allowed to convert the DataFrame into a Numpy array for
    faster computation.
    """
    similarity_dict = {}
    # TODO: Handle the absence of ratings (missing values in the matrix)

    # It seems this has to be ignored for the calculation because /0 issue
    # I could set it to 0, and treat it as a vector from the origin, I don't see any nans in the data though
    # what if their rating is 0, that just removes muddys the data
    # They would be incredibly similar then because 93% of data would be 0
    # apparently there's like 1,486,000 out of 1,586,000 so like 93% empty
    # ahh its being pivoted.

    # print(f'Before there are {matrix.isnull().values.sum()} missing values.')
    # row_check = matrix.isnull().values.all(axis=0).sum()
    # col_check = matrix.isnull().values.all(axis=1).sum()
    # print(f'There are {row_check} rows that are all NaN and {col_check} columns that are all NaN.')
    # matrix = matrix.fillna(0)
    # print(f'After there are {matrix.isnull().values.sum()} missing values.')

    # I think I need to mask?
    # Ahh I think its just comparing parallel vectors which makes it simpler

    # is this a quadratic speed problem? I'm trying to think of a better approach
    # I could do linear time if they're treated as 0s and not comparing masks between each row..
    # I could make it a bit quicker if for i total rows: for j in [i:total rows]: no extra checks. lower triangle matrix

    # I think I might be on the wrong approach because the TO DO here is just asking for data filtering not axis
    # what if I impute 0 and then just softmax essentially, okay I just need to do something

    #wait what If i just leave it nan? can't lol they propagate throught the calc
    matrix = matrix.fillna(0)

    # TODO: If axis is 1, what do you need TODO to calculate the similarity
    # between items (columns)
    if axis == 1:
        #just flip it
        matrix = matrix.T
        pass

    # TODO: loop through each couple of entities to calculate their cosine
    # similarity and store these results
    total_rows = matrix.shape[0]
    sim_matrix = []
    for row in range(0,total_rows):
        vector = matrix.iloc[[row]]
        #np.sqrt is slow... could just square everything maybe idk not bothered
        mag_1 = np.linalg.norm(vector)
        mag_2 = np.linalg.norm(matrix,axis=1) #doesn't matter because its preflipped
        similarity_vector = np.dot(vector, matrix.T) / (mag_1 * mag_2)
        sim_matrix.append(similarity_vector.flatten())
    sim_matrix =np.array(sim_matrix)

    #no matplotlib so cant check the image lol


    for row in range(0,total_rows):
        sim_matrix[row,row] = np.nan
        row_vec = sim_matrix[row,:]
        top_k_indicies_list = sorted(range(len(row_vec)), key=lambda sub: row_vec[sub],reverse=True)[-k:]
        top_k_similarity_list = [row_vec[i] for i in top_k_indicies_list]
        similarity_dict[row] = {'top_similar_indices':top_k_indicies_list,
                                'top_similar_values':top_k_similarity_list}

    # TODO: sort the similarity scores for each entity and add the top k most
    # similar entities to the similarity_dict

    return similarity_dict


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictonary containing user similarities, \
            obtained using the similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to user-based \
        collaborative filtering
    """
    # TODO: retrieve the topk most similar users for the target user

    # TODO: implement user-based collaborative filtering according to the
    # formula discussed during the lecture (reported in the PDF attached to
    # the assignment)
    numerator = 0
    denominator = 0

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, \
            obtained using the similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to item-based
        collaborative filtering
    """
    # TODO: retrieve the topk most similar users for the target item

    # TODO: implement item-based collaborative filtering according to the
    # formula discussed during the lecture (reported in the PDF attached to
    # the assignment)
    numerator = 0
    denominator = 0

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,
        learning_rate=0.001,
        regularization=0.02,
        n_steps=2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function should contain the code to implement matrix factorisation
    using the Gradient Descent with Regularization method (according to the psuedo code
    seen at the lecture), returning the user and item matrices.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        feature_dimension (int): number of latent features (default=2)
        learning_rate (float): learning rate for gradient descent \
            (default=0.001)
        regularization (float): regularization parameter (default=0.02)
        n_steps (int): number of iterations for gradient descent \
            (default=2000)

    Returns:
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
    """

    user_matrix = np.random.rand(utility_matrix.shape[0], feature_dimension)
    item_matrix = np.random.rand(utility_matrix.shape[1], feature_dimension)

    for step in range(n_steps):
        # TODO: Implement the algorithm to update user_matrix and item_matrix
        pass

    return user_matrix, item_matrix


if __name__ == "__main__":
    path = "../Data/u.data"
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])

    df = df.pivot_table(
        index='UserID',
        columns='MovieID',
        values='Rating'
    )

    # You can use this section for testing the similarity_matrix function:
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3, []))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10, []))

    # You can use this section for testing the user_based_cf and the
    # item_based_cf functions: Return the predicted ratings assigned by user
    # 13 to movie 100:
    user_id = 13
    movie_id = 100

    u_predicted_rating = user_based_cf(
        user_id,
        movie_id,
        user_similarity_matrix,
        user_item_matrix=df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id,
        item_similarity_matrix,
        user_item_matrix=df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
        [5, 2, 4, 4, 3],
        [3, 1, 2, 4, 1],
        [2, np.nan, 3, 1, 4],
        [2, 5, 4, 3, 5],
        [4, 4, 5, 4, np.nan],
    ])
    user_matrix, item_matrix = matrix_factorization(
        utility_matrix, learning_rate=0.001, n_steps=5000
    )

    print("Current guess:\n", np.dot(user_matrix, item_matrix.T))
