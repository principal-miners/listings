import os
import ast
import seaborn as sns
import pandas as pd
import math
import numpy as np
import pandas

from sklearn.preprocessing import LabelEncoder

from Utils.DataUtils import *

# modify data_path to point to the folder where listings.csv is.
data_path = "C:\\Users\\SSrih\\OneDrive\\UChicago\\DataMining\\project\\NYData"
# data_path = "C:\\GitHub\\listings\\data"
listings_path = os.path.join(data_path, "listings.csv")
kmeans_topcs_path = os.path.join(data_path, "kmeans_topics.csv")

PERCENTILE_CROP = [1, 98]


def null(df, name):
    print('null: ',df[name].isnull().sum())
    print('null percent: ', df[name].isnull().sum()/len(df))
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
    # HARDCODE AS 1 AND 0 FOR T AND F RESPECTIVELY
    di = {"t": 1, "f": 0}
    listings["host_is_superhost"].replace(di, inplace=True)

    # host_has_profile_pic
    listings["host_has_profile_pic"].replace(di, inplace=True)

    # host_identity_verified
    listings["host_identity_verified"].replace(di, inplace=True)

    # is_location_exact
    listings["is_location_exact"].replace(di, inplace=True)

    # instant_bookable
    listings["instant_bookable"].replace(di, inplace=True)

    # require_guest_profile_picture
    listings["require_guest_profile_picture"].replace(di, inplace=True)

    # require_guest_phone_verification
    listings["require_guest_phone_verification"].replace(di, inplace=True)

    # -----------------------------------------------------------------------------------------------------------------#
    # Srihari's pipeline
    # -----------------------------------------------------------------------------------------------------------------#

    cols_to_ohe = ["host_response_time", "neighbourhood_group_cleansed", "property_type",
                   "room_type", "bed_type", "cancellation_policy"]
    listings = pd.get_dummies(listings, columns=cols_to_ohe, drop_first=True)

    return listings


def remove_outliers(listings):
    percentiles = list(range(0, 101, 1))
    price_percentile = {}
    for p in percentiles:
        price_percentile[p] = np.percentile(listings['price'].values, p)

    price_percentile = pd.DataFrame.from_dict(price_percentile, orient='index')

    listings = listings[listings["price"] <= price_percentile.iloc[PERCENTILE_CROP[1], :].values[0]]
    listings = listings[listings["price"] >= price_percentile.iloc[PERCENTILE_CROP[0], :].values[0]]

    return listings


def round_price(listings):
    base = 5

    def roundto(row):
        return int(base * round(float(row) / base))
    listings["price"] = listings["price"].apply(roundto)

    return listings


def handle_missing_values(listings):

    # -----------------------------------------------------------------------------------------------------------------#
    # Monu's pipeline
    # -----------------------------------------------------------------------------------------------------------------#

    # First drop rows that cant be used
    rows_to_drop = listings[listings['host_listings_count'] != listings['host_total_listings_count']].index
    listings.drop(index=rows_to_drop, axis=0, inplace=True)
    listings.drop(labels="host_listings_count", axis=1)

    # Host response time
    listings['host_response_time'] = listings['host_response_time'].fillna('na')

    # Host response rate
    listings['host_response_rate'] = listings['host_response_rate'].fillna(str(int(get_hrr_fillval(listings))))
    listings['host_response_rate'] = listings['host_response_rate'].str.strip('%').astype('int')

    # host_is_superhost
    listings['host_is_superhost'] = listings['host_is_superhost'].fillna('f')

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

    # price
    listings['price'] = listings['price'].str.strip('').str.strip('$').str.replace(',', '').astype('float')
    # Round the price column to multiple of nearest 5$
    round_price(listings)

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

    # cancellation_policy
    listings['cancellation_policy'].loc[listings['cancellation_policy'].str.contains('strict')] = 'strict'
    listings['cancellation_policy'].loc[listings['cancellation_policy'].str.contains('long_term')] = 'strict'

    # reviews_per_month
    listings['reviews_per_month'] = listings['reviews_per_month'].fillna(listings['reviews_per_month'].median())

    # -----------------------------------------------------------------------------------------------------------------#
    # Srihari's pipeline
    # -----------------------------------------------------------------------------------------------------------------#

    # descrption and host_about
    listings["description"].fillna("", inplace=True)
    listings["host_about"].fillna("", inplace=True)
    listings["description"] = listings["description"].astype(str)
    listings["host_about"] = listings["host_about"].astype(str)

    listings["desc_len"] = listings["description"].apply(len)
    listings["host_about_len"] = listings["host_about"].apply(len)

    # review_scores
    review_scores_cols = ['review_scores_value', 'review_scores_location', 'review_scores_checkin',
                          'review_scores_cleanliness', 'review_scores_communication', 'review_scores_accuracy',
                          'review_scores_rating']
    for col in review_scores_cols:
        listings[col].fillna(listings[col].median(), inplace=True)

    # first_review and last_review
    listings['last_review'] = pd.to_datetime(listings['last_review'])
    listings['lreview_year'] = listings['last_review'].dt.year
    listings['lreview_month'] = listings['last_review'].dt.month
    listings['lreview_day'] = listings['last_review'].dt.day
    listings['first_review'] = pd.to_datetime(listings['first_review'])
    listings['freview_year'] = listings['first_review'].dt.year
    listings['freview_month'] = listings['first_review'].dt.month
    listings['freview_day'] = listings['first_review'].dt.day

    # Fill the missing values with zeros instead of an unknown category
    # listings["first_review"].fillna("XX", inplace=True)
    # listings["last_review"].fillna("XX", inplace=True)
    listings['lreview_year'].fillna(0, inplace=True)
    listings['lreview_month'].fillna(0, inplace=True)
    listings['lreview_day'].fillna(0, inplace=True)
    listings['freview_year'].fillna(0, inplace=True)
    listings['freview_month'].fillna(0, inplace=True)
    listings['freview_day'].fillna(0, inplace=True)

    # Get the number of days between reviews
    listings['ndays_between_f_l_reviews'] = abs(listings['lreview_day'] - listings['freview_day'])

    # Get number of days host has been with AirBnb
    listings['host_since'] = pd.to_datetime(listings['host_since'])
    listings['calendar_last_scraped'] = pd.to_datetime(listings['calendar_last_scraped'])
    listings["ndays_host"] = (listings["calendar_last_scraped"] - listings["host_since"]).dt.days

    listings["ndays_last_review"] = (listings["calendar_last_scraped"] - listings["last_review"]).dt.days
    listings['ndays_last_review'].fillna(999999, inplace=True)

    # Amenities
    def get_num_amenities(row):
        a = row[1:-1].split(",")
        return len(a)

    listings["num_amenities"] = listings["amenities"].apply(get_num_amenities)

    # Hack, Drop any remaining rows that have nulls
    listings.drop(labels=["last_review", "first_review", "calendar_last_scraped", "host_since",
                          "amenities", "description", "host_about"], axis=1, inplace=True)

    listings.dropna(inplace=True)

    return listings


def remove_corr_cols(listings):
    cols_to_drop = ["host_total_listings_count",
                    "minimum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights",
                    "maximum_nights_avg_ntm", "availability_60", "availability_90", "availability_365",
                    "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
                    "calculated_host_listings_count_shared_rooms", "freview_year",
                    "maximum_minimum_nights", "minimum_nights_avg_ntm", "number_of_reviews_ltm"]

    for col in cols_to_drop:
        if col in listings.columns:
            listings.drop(labels=col, inplace=True, axis=1)

    return listings


def main():
    listings = pd.read_csv(listings_path)

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
                    'license',
                    'state',
                    'is_business_travel_ready',
                    'square_feet',
                    'weekly_price',
                    'monthly_price',
                    "city",
                    "market",
                    "host_location",
                    "host_neighbourhood",
                    "zipcode",
                    "neighbourhood",
                    "neighbourhood_cleansed",
                    "state",
                    "smart_location",
                    "country_code",
                    "country",
                    "latitude",
                    "longitude",
                    "calendar_updated",
                    "has_availability",
                    "requires_license"
                    ]

    for col in cols_to_drop:
        if col in listings.columns:
            listings.drop(labels=col, inplace=True, axis=1)

    # Note, there are some columns with user-input data that are not dropped
    # To drop these columns, simply uncomment the next two lines of code.
    nlp_cols = ['jurisdiction_names',
                 'notes',
                 'interaction',
                 'access',
                 'house_rules',
                 'neighborhood_overview',
                 'transit',
                 'space',
                 'summary',
                 'name']
    for col in nlp_cols:
        if col in listings.columns:
            listings.drop(labels=col, inplace=True, axis=1)

    # 1. Handle missing values
    listings = handle_missing_values(listings)
    print("1", listings.shape)

    # 2. Remove outliers
    listings = remove_outliers(listings)
    print("2", listings.shape)

    # 2. Encode variables
    listings = encode_variables(listings)
    print("3", listings.shape)

    # Save the dataframe to disk
    out_path = os.path.join(data_path, "cleaned_listings_with_outliers.csv")
    print(out_path)
    listings.to_csv(out_path, index=False)
    print("4", listings.shape)
    exit(12)

    # 3. Join the KMeans topics file
    kmeans_topics = pd.read_csv(kmeans_topcs_path)
    combined_table = pd.merge(left=listings, right=kmeans_topics,
                              left_on="id", right_on="listing_id", how="right")
    if "listing_id" in combined_table.columns:
        combined_table.drop(labels=["listing_id"], axis=1, inplace=True)

    # Save the dataframe to disk
    out_path = os.path.join(data_path, "cleaned_with_nlp_listings.csv")
    print(out_path)
    combined_table.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
