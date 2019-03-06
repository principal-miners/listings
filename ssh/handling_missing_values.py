import os
import ast
import seaborn as sns
import pandas as pd
import math
import numpy as np
import pandas

from sklearn.preprocessing import LabelEncoder

from Utils.UtilsViz import *
from Utils.DataUtils import *

# data_path = Used to save the result file
data_path = os.path.join("/home/srihari/Projects/DataMining/Project", "Data")

# ny_datapath = path to listings.csv
ny_datapath = os.path.join(data_path, "NY")


def null(df, name):
    print('null: ',df[name].isnull().sum())
    print('null percent: ',df[name].isnull().sum()/len(df))
    print(df[name].value_counts())
    print(df[name].dtype)


def encode(df, name):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df[name].unique())
    values = le.transform(df[name].values)
    df[name] = values
    # print(df[name].value_counts())


def get_hrr_fillval(listings):
    non_null = listings['host_response_rate'].dropna(axis=0)
    fv = non_null.str.strip('%').astype('int').median()
    return fv


def num_nan_rows(df):
    return df.shape[0] - df.dropna().shape[0]


def encode_variables(listings):

    # -----------------------------------------------------------------------------------------------------------------#
    # Monu's pipeline
    # -----------------------------------------------------------------------------------------------------------------#

    # host_is_superhost
    le = LabelEncoder()
    le.fit(listings['host_is_superhost'].unique())
    values = le.transform(listings['host_is_superhost'].values)
    listings['host_is_superhost'] = values

    # host_has_profile_pic
    le = LabelEncoder()
    le.fit(listings['host_has_profile_pic'].unique())
    values = le.transform(listings['host_has_profile_pic'].values)
    listings['host_has_profile_pic'] = values

    # host_identity_verified
    le = LabelEncoder()
    le.fit(listings['host_identity_verified'].unique())
    values = le.transform(listings['host_identity_verified'].values)
    listings['host_identity_verified'] = values

    # is_location_exact
    le = LabelEncoder()
    le.fit(listings['is_location_exact'].unique())
    values = le.transform(listings['is_location_exact'].values)
    listings['is_location_exact'] = values

    # instant_bookable
    encode(listings, 'instant_bookable')

    # require_guest_profile_picture
    encode(listings, 'require_guest_profile_picture')

    # require_guest_phone_verification
    encode(listings, 'require_guest_phone_verification')

    # Incase we also need OHE, uncomment line below
    # listings = pd.get_dummies(listings)

    return listings


def handle_missing_values(listings):

    # -----------------------------------------------------------------------------------------------------------------#
    # Monu's pipeline
    # -----------------------------------------------------------------------------------------------------------------#

    # First drop rows that cant be used
    rows_to_drop = listings[listings['host_listings_count'] != listings['host_total_listings_count']].index
    listings.drop(index=rows_to_drop, axis=0, inplace=True)

    # Host response time
    listings['host_response_time'] = listings['host_response_time'].fillna('null')

    # Host response rate
    listings['host_response_rate'] = listings['host_response_rate'].fillna(str(int(get_hrr_fillval(listings))))
    listings['host_response_rate'] = listings['host_response_rate'].str.strip('%').astype('int')

    # host_is_superhost
    listings['host_is_superhost'] = listings['host_is_superhost'].fillna('f')

    # host_listings_count
    listings.drop(['host_listings_count'], axis=1, inplace=True)

    # host_verifications
    listings['host_verifications'] = listings['host_verifications'].apply(lambda x: ast.literal_eval(x))
    listings['host_verifications_count'] = listings['host_verifications'].fillna('').apply(lambda x: len(x))
    listings.drop(['host_verifications'], axis=1, inplace=True)

    listings['host_identity_verified'].value_counts()

    # property_type
    strings = ("Apartment", "House", "Townhouse", "Loft", "Condominium", "Serviced apartment")
    apartment_list = list([])
    for line in listings['property_type']:
        if any(s in line for s in strings):
            apartment_list.append('yes')
        else:
            apartment_list.append('no')

    listings['prop'] = apartment_list
    listings.loc[listings['prop'] == 'no', 'property_type'] = 'Other'
    listings.loc[listings['property_type'] == 'Houseboat', 'property_type'] = 'Other'
    listings.drop(['prop'], axis=1, inplace=True)

    # bathroom_type
    listings['bathrooms'] = listings['bathrooms'].fillna(1.0).astype('float')

    # bedrooms
    listings['bedrooms'] = listings['bedrooms'].fillna(1.0).astype('float')

    # beds
    listings['beds'] = listings['beds'].fillna(1.0).astype('float')

    # square_feet
    listings.drop(['square_feet'], axis=1, inplace=True)

    # price
    listings['price'] = listings['price'].str.strip('').str.strip('$').str.replace(',', '').astype('float')

    # weekly_prie
    listings.drop(['weekly_price'], axis=1, inplace=True)

    # monthly_price
    listings.drop(['monthly_price'], axis=1, inplace=True)

    # security_deposit
    secdrp_fillval = \
        listings['security_deposit'].dropna(axis=0).str.strip('$').str.replace(',', '').astype('float').median()
    listings['security_deposit'] = \
        listings['security_deposit'].fillna(str(secdrp_fillval)).str.strip('$').str.replace(',', '').astype('float')

    # cleaning_fee
    cleanfee_fillval =\
        listings['cleaning_fee'].dropna(axis=0).str.strip('$').str.replace(',', '').astype('float').median()
    listings['cleaning_fee'] = \
        listings['cleaning_fee'].fillna(str(cleanfee_fillval)).str.strip('$').str.replace(',', '').astype('float')

    # extra_people
    listings['extra_people'] = listings['extra_people'].str.strip('$').str.replace(',', '').astype('float')

    # is_business_travel_ready
    listings.drop(['is_business_travel_ready'], axis=1, inplace=True)

    # cancellation_policy
    listings['cancellation_policy'].loc[listings['cancellation_policy'].str.contains('strict')] = 'strict'
    listings['cancellation_policy'].loc[listings['cancellation_policy'].str.contains('long_term')] = 'strict'

    # reviews_per_month
    listings['reviews_per_month'] = listings['reviews_per_month'].fillna(listings['reviews_per_month'].median())

    # -----------------------------------------------------------------------------------------------------------------#
    # Srihari's pipeline
    # -----------------------------------------------------------------------------------------------------------------#

    # review_scores
    review_scores_cols = ['review_scores_value', 'review_scores_location', 'review_scores_checkin',
                          'review_scores_cleanliness', 'review_scores_communication', 'review_scores_accuracy',
                          'review_scores_rating']
    for col in review_scores_cols:
        listings[col].fillna(listings[col].median(), inplace=True)

    # city
    listings["city"].fillna("NYC", inplace=True)

    # market
    geo_cols = ["city", "neighbourhood", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "market",
                "host_neighbourhood"]

    no_geonan_listings = listings[geo_cols].dropna()
    map_nbc_market = {}
    for nbc in no_geonan_listings["neighbourhood_cleansed"].unique():
        map_nbc_market[nbc] = listings[listings["neighbourhood_cleansed"] == nbc]["market"].unique()

    def impute_market(row):
        if type(row["market"]) != type("a"):
            nb_cleansed = row["neighbourhood_cleansed"]
            val = map_nbc_market[nb_cleansed][0]
            return val
        return row

    listings["market"] = listings[["market", "neighbourhood_cleansed"]].apply(impute_market, axis=1)

    # host_location
    listings["host_location"].fillna("XX", inplace=True)

    # host_neighbourhood
    listings["host_neighbourhood"].fillna("XX", inplace=True)

    # neighbourhood
    listings["neighbourhood"].fillna(listings["neighbourhood_cleansed"], inplace=True)

    # zipcode
    listings["zipcode"].fillna("XX", inplace=True)

    # first_review and last_review
    listings['last_review'] = pd.to_datetime(listings['last_review'])
    listings['lreview_year'] = listings['last_review'].dt.year
    listings['lreview_month'] = listings['last_review'].dt.month
    listings['lreview_day'] = listings['last_review'].dt.day
    listings['first_review'] = pd.to_datetime(listings['first_review'])
    listings['freview_year'] = listings['first_review'].dt.year
    listings['freview_month'] = listings['first_review'].dt.month
    listings['freview_day'] = listings['first_review'].dt.day

    listings["first_review"].fillna("XX", inplace=True)
    listings["last_review"].fillna("XX", inplace=True)
    listings['lreview_year'].fillna("XX", inplace=True)
    listings['lreview_month'].fillna("XX", inplace=True)
    listings['lreview_day'].fillna("XX", inplace=True)
    listings['freview_year'].fillna("XX", inplace=True)
    listings['freview_month'].fillna("XX", inplace=True)
    listings['freview_day'].fillna("XX", inplace=True)

    return listings


def main():
    listings = pd.read_csv(os.path.join(ny_datapath, "listings.csv"))

    # 0. Drop some columns
    cols_to_drop = ['listing_url',
                    'scrape_id',
                    'last_scraped',
                    'experiences_offered',
                    'thumbnail_url',
                    'medium_url',
                    'picture_url',
                    'xl_picture_url',
                    'host_id',
                    'host_url',
                    'host_name',
                    'host_acceptance_rate',
                    'host_thumbnail_url',
                    'host_picture_url',
                    'street',
                    'license']
    listings.drop(labels=cols_to_drop, inplace=True, axis=1)

    # Note, there are some columns with user-input data that are not dropped
    # To drop these columns, simply uncomment the next two lines of code.
    # nlp_cols = ['jurisdiction_names',
    #              'notes',
    #              'interaction',
    #              'access',
    #              'house_rules',
    #              'neighborhood_overview',
    #              'host_about',
    #              'transit',
    #              'space',
    #              'summary',
    #              'name',
    #              'description']
    # listings.drop(labels=nlp_cols, inplace=True, axis=1)

    # 1. Handle missing values
    listings = handle_missing_values(listings)

    # 2. Encode variables
    listings = encode_variables(listings)

    # Save the dataframe to disk
    out_path = os.path.join(data_path, "cleaned_listings.csv")
    listings.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
