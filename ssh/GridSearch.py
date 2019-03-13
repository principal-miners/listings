import os
import seaborn as sns
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as MSE

from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42


def main():
    data_path = os.path.join(os.getcwd(), "../data/cleaned_listings.csv")
    listings = pd.read_csv(data_path, index_col="id")
    PERCENTILE_CROP = [1, 98]
    print(listings.shape)

    percentiles = list(range(0, 101))
    price_percentile = {}
    for p in percentiles:
        price_percentile[p] = np.percentile(listings['price'].values, p)

    price_percentile = pd.DataFrame.from_dict(price_percentile, orient='index')

    listings_filtered = listings[listings["price"] <= price_percentile.iloc[PERCENTILE_CROP[1], :].values[0]]
    listings_filtered = listings_filtered[
    listings_filtered["price"] >= price_percentile.iloc[PERCENTILE_CROP[0], :].values[0]]

    base = 5

    def roundto(row):
        return int(base * round(float(row) / base))

    # listings_filtered["price"] = listings_filtered["price"].apply(roundto)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_SEED)
    train, test = train_test_split(listings_filtered, test_size=0.2)

    def check_rep(row):
        if (row <= 200) | (row == 250) | (row == 350) | (row == 450) | (row == 550):
            return 0
        elif (row > 200) & (row < 300) & (row != 250):
            return 1
        elif (row > 300) & (row < 400) & (row != 350):
            return 2
        elif (row > 400) & (row < 500) & (row != 450):
            return 3
        else:
            return 4

    train["flag_ur"] = train["price"].apply(check_rep)

    # -------------------------------------------------------------------------------------------------------- #
    vcs = train["flag_ur"].value_counts()

    ycol = ["flag_ur"]
    xcol = [i for i in train.columns if i not in ycol]

    x = train[xcol].values
    y = train[ycol].values

    smote_sampling_strategy = {
        1: int(vcs[1] * 4)
        , 2: int(vcs[1] * 2)
        , 3: int(vcs[1] * 2)
        , 4: int(vcs[1] * 2)
    }
    sm = SMOTE(random_state=RANDOM_SEED, sampling_strategy=smote_sampling_strategy, n_jobs=-1)

    x_new, y_new = sm.fit_sample(x, y)

    # -------------------------------------------------------------------------------------------------------- #
    train.drop(labels=["flag_ur"], axis=1, inplace=True)

    # Get the index of the price columns
    def get_index(vallist, val):
        return vallist.index(val)

    price_index = get_index(list(train.columns), "price")

    y = x_new[:, price_index]
    x_train = np.delete(x_new, price_index, axis=1)

    y_new = []
    for elem in y:
        y_new.append(roundto(elem))

    y_train = y_new
    # -------------------------------------------------------------------------------------------------------- #
    # Standardisation
    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_train)

    # Get the index of the price columns
    def get_index(vallist, val):
        return vallist.index(val)

    price_index = get_index(list(test.columns), "price")

    x_tmp = test.values
    y_test = x_tmp[:, price_index]
    x_test = np.delete(x_tmp, price_index, axis=1)

    y_tmp = []
    for elem in y_test:
        y_tmp.append(roundto(elem))

    y_test = y_tmp

    x_test = standard_scaler.transform(x_test)

    # -------------------------------------------------------------------------------------------------------- #
    # Grid Search

    rfr = RandomForestRegressor()

    # Instantiate the GridSearchCV object and run the search
    parameters = {
        'criterion': ['mse'],
        'n_estimators': [150, 200],
        'max_features': [None, 'sqrt'],
        'random_state': [42]
    }

    searcher = GridSearchCV(rfr, parameters, n_jobs=6)
    searcher.fit(x_train, y_train)

    # Report the best parameters
    print("Best CV RF params", searcher.best_params_)


if __name__ ==  "__main__":
    main()
