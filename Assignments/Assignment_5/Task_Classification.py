# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from ucimlrepo import fetch_ucirepo 

# Import data From UCI repository about Portuguese wine qualities:
# This dataset contains features related to wine properties and the labels 
# related to the different wine qualities. Our goal is to predict the correct
# wine quality just looking at the features and not at the labels. 
wine_quality = fetch_ucirepo(id=186) 
  
X = wine_quality.data.features 
y = wine_quality.data.targets 

# Define the function for Random Forest classification with cross-validation
def random_forest_experiment(n_estimators, min_samples_leaf, X, y):
    """
    Trains a RandomForestClassifier with specified parameters and performs 10-fold cross-validation.
    
    Parameters:
    - n_estimators: Number of trees in the forest.
    - min_samples_leaf: Minimum number of samples required to be at a leaf node.
    - X: Feature matrix (wine features).
    - y: Target vector (wine quality labels).
    
    Returns:
    - Mean cross-validation accuracy (measure to estimate the accuracy of our prediction
      based on a 10-fold cross validation)
    """
    # TODO: Create a RandomForestClassifier model with given parameters
    # n_estimators and min_sample_leaf as input, set random_state to 2024

    forest = RandomForestClassifier(n_estimators=n_estimators,
                                    min_samples_leaf=min_samples_leaf,
                                    # oob_score=True,
                                    random_state=2024,
                                    )
    # TODO: Implement 10-fold cross-validation with random_state set to 2024
    # and compute the scores based on cross-validation accuracy
    kf = KFold(n_splits=10, random_state=2024,shuffle=True)

    score = cross_val_score(forest,
                            X,
                            y,
                            cv=kf,
                            scoring='accuracy')

    # Return the average cross-validation accuracy
    return score.mean()

# Function to return the best result based on hyperparameters
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    """
    Finds the best hyperparameter combination based on cross-validation accuracy.
    
    Parameters:
    - n_estimators_list: List of different values for the number of trees.
    - min_samples_leaf_list: List of different values for min_samples_leaf.
    - X: Feature matrix (wine features).
    - y: Target vector (wine quality labels).
    
    Returns:
    - Best hyperparameter combination and corresponding accuracy.
    """
    results = []

    # TODO: Iterate over all combinations of hyperparameters and calculate the 
    # cross-validation accuracy by using random_forest_experiment function, append 
    # n_estimators, min_sample_leaf and accuracy to results.

    #small amount of factors, quick run time, 16 runs is okay
    n=0
    for n_estimators in n_estimators_list:
        for min_samples_leaf in min_samples_leaf_list:
            score = random_forest_experiment(n_estimators=n_estimators,
                                             min_samples_leaf=min_samples_leaf,
                                             X=X,
                                             y=y)
            results.append((n_estimators, min_samples_leaf, score))
            n+=1
            print(f'Model {n}, with a score of {score}')



    best_score = 0
    best_index = 0
    for index, model in enumerate(results):
        if model[2] > best_score:
            best_score = model[2]
            best_index = index

    best_result = results[best_index]
    # Return the best combination of hyperparameters and the corresponding accuracy
    return best_result

    # TO DO: After running the above function, manually input your best result here
def manually_entered_best_params_and_accuracy():
    best_params = {'n_estimators': 500, 'min_samples_leaf': 500}  # Example, to be replaced with the retrieved best parameters
    best_accuracy = 0.9999  # Example accuracy, to be replaced with your achieved accuracy
    return best_params, best_accuracy
    
# Main function
if __name__ == "__main__":
    # Experiment with different values for n_estimators and min_samples_leaf
    # to find the best parameters setting among the following options:
    n_estimators_list = [10, 50, 10, 10]
    min_samples_leaf_list = [1, 10, 50, 100]

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y.values.ravel())

    print(f"Best Hyperparameters: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best Accuracy: {best_params['accuracy']:.4f}")
