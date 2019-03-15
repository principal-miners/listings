import os
import seaborn as sns
import pandas as pd
import numpy as np
import math

from scipy.stats import randint as sp_randint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import mean_squared_error as MSE

from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def main():
    data_path = os.path.join(os.getcwd(), "../data/cleaned_listings.csv")
    listings = pd.read_csv(data_path, index_col="id")

    base = 5

    def roundto(row):
        return int(base * round(float(row) / base))

    def check_rep(row):
        if (row <= 200) | (row == 250) | (row == 350) | (row == 450) | (row == 550):
            return 0
        elif (row > 200) & (row < 300) & (row != 250):
            return 1
        elif (row > 300) & (row < 400) & (row != 350):
            return 2
        else:
            return 3

    listings["flag_ur"] = listings["price"].apply(check_rep)

    # -------------------------------------------------------------------------------------------------------- #
    vcs = listings["flag_ur"].value_counts()
    ycol = ["flag_ur"]
    xcol = [i for i in listings.columns if i not in ycol]

    x = listings[xcol].values
    y = listings[ycol].values

    smote_sampling_strategy = {
        1: int(vcs[1] * 2)
        , 2: int(vcs[2] * 2)
        , 3: int(vcs[3] * 2)
    }
    sm = SMOTE(random_state=RANDOM_SEED, sampling_strategy=smote_sampling_strategy, n_jobs=-1)
    # Fit the smote onto the sample
    x_new, y_new = sm.fit_sample(x, y)

    # Drop the flag column
    listings.drop(labels=["flag_ur"], axis=1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------
    # Overwrite X and Y
    # Get the index of the price columns
    def get_index(vallist, val):
        return vallist.index(val)
    price_index = get_index(list(listings.columns), "price")

    y = x_new[:, price_index]
    x = np.delete(x_new, price_index, axis=1)
    for i in range(len(y)):
        y[i] = roundto(y[i])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=RANDOM_SEED)

    # -------------------------------------------------------------------------------------------------------- #
    # Standardisation
    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_train)
    x_test = standard_scaler.transform(x_test)

    # ======================================================================================================== #
    # GBR
    # ======================================================================================================== #
    # Random Search
    gbr = GradientBoostingRegressor()

    param_dist = {"n_estimators": [400],
                  "random_state": [42],
                  "max_depth": [5, 10, 20, 40],
                  "max_features": ["sqrt", "log2"],
                  "min_samples_split": list(range(5, x_train.shape[1], 5)),
                  "criterion": ["mse"],
                  "min_samples_leaf": [2, 3, 5],
                  "learning_rate": [0.5, 0.1],
                  "subsample": [0.8, 1, 0.7]
                  }

    n_iter_search = 2
    random_search = RandomizedSearchCV(gbr, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, n_jobs=6)

    random_search.fit(x_train, y_train)
    report(random_search.cv_results_)

    print("Best CV RF params", random_search.best_params_)

    # ======================================================================================================== #
    # Random forest
    # ======================================================================================================== #
    # Random Search
    rfr = RandomForestRegressor()

    param_dist = {"n_estimators": [400],
                   "max_depth": [20, 40, None],
                   "max_features": ["sqrt", "log2", None],
                   "min_samples_split": list(range(5, x_train.shape[1], 5)),
                   "bootstrap": [True, False],
                   "criterion": ["mse"],
                   "random_state": [42]
                  }

    n_iter_search = 20
    random_search = RandomizedSearchCV(rfr, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, n_jobs=6)

    random_search.fit(x_train, y_train)
    report(random_search.cv_results_)

    # -------------------------------------------------------------------------------------------------------- #
    # Grid Search

    rfr = RandomForestRegressor()

    # Instantiate the GridSearchCV object and run the search
    parameters = {
        "n_estimators": [400],
        "max_depth": [80, 100],
        "max_features": ["sqrt"],
        "min_samples_split": [5, 8, 10],
        "bootstrap": [False],
        "criterion": ["mse"],
        "random_state": [42]
    }

    searcher = GridSearchCV(rfr, parameters, n_jobs=6)
    searcher.fit(x_train, y_train)

    report(searcher.cv_results_)

    # Report the best parameters
    print("Best CV RF params", searcher.best_params_)


if __name__ ==  "__main__":
    main()
