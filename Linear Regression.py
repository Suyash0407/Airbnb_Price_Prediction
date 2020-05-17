"""
Problem Statement -  To find the price of a house based on various factors
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path = os.getcwd()
%time df = pd.read_csv(path+'london_listing.csv')

##EDA

#Dataframe shape
df.shape

#Dataframe info
df.info()

#Remove duplicates
df.drop_duplicates(keep='first', inplace = True)


#Segregate numeric and categorical data
df_numeric_data = df._get_numeric_data()
df_categorical_data = df[list(set(df.columns) - set(df._get_numeric_data().columns))]

#-------------------Naive Model 1------------------------
cat_columns = df_categorical_data.columns

for col in cat_columns :
    df_categorical_data[col] = df_categorical_data[col].astype('category').cat.codes
    
df_modified = pd.concat([df_numeric_data, df_categorical_data], axis = 1)

##Check for nulls and replace with 0
df_modified.isnull().sum()
df_modified.fillna(0, inplace = True)

X = df_modified.drop('price', axis = 1)
y = df_modified['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Metrics
from sklearn.metrics import mean_absolute_error, r2_score

print('RMSE : ', np.sqrt(mean_absolute_error(y_test, y_pred)))
print('R2 score : ', r2_score(y_test, y_pred))



#--------------------------EDA--------------------------

#Describe numerical data
df_numeric_data.describe()

#Describe categorical data
df_categorical_data.describe()

#Correlation matrix
corr = df.corr()
#Set figure size
plt.figure(figsize=(20,10))
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

#Dataframe
df.head()


#-----------------Categorical Data EDA---------------------#


df_categorical_data.head()   
 
df_categorical_data.describe()

#Check nulls
df_categorical_data.isnull().sum()

#target variable
df_categorical_data['price'].isnull().sum()

df_categorical_data['price'].value_counts()

#Removing features assumed to be not very relavant to determine house rent
temp = df_categorical_data.head()
df_categorical_data.columns

categories_to_drop = ['host_has_profile_pic', 'neighborhood_overview', 'notes', 
                     'picture_url','first_review', 'host_since', 'summary', 
                     'interaction', 'neighbourhood', 'description',
                     'require_guest_profile_picture', 'name', 'access', 
                     'jurisdiction_names', 'host_thumbnail_url', 'transit',
                     'host_location', 'host_response_rate', 'host_url',
                     'experiences_offered', 'require_guest_phone_verification',
                     'space', 'last_review', 'host_neighbourhood', 'last_scraped',
                     'house_rules', 'host_picture_url', 'instant_bookable',
                     'requires_license', 'has_availability', 'host_name', 
                     'calendar_last_scraped', 'host_response_time', 
                     'host_is_superhost', 'host_verifications',
                     'host_about', 'listing_url', 'calendar_updated', 'license',
                     'is_business_travel_ready', 'cancellation_policy', 
                     'host_identity_verified', 'is_location_exact', 
                     'security_deposit', 'extra_people']
df_categorical_data.drop(categories_to_drop, axis = 1, inplace= True)

#Check unique values for columns
for col in list(df_categorical_data.columns) :
    print('unique values in {}: {}'.format(col, 
          len(np.unique(list(df_categorical_data[col])))))
    

#Converting price values from string to float
# =============================================================================
# 
# =============================================================================
# 1. Covertiving price to float
price = []
for p in list(df_categorical_data['price']) :
    p = str(p).replace(',', '' )
    price.append(float(p[0:]))

df1 =  df_categorical_data['price']
df_numeric_data['price'] = price
df_categorical_data.drop('price', axis = 1, inplace = True)

# 2. Covertiing cleaning fee to float
df_categorical_data['cleaning_fee'].isnull().sum()


price = []
# Cleaning_fee consists of nan values that needs to be handled
for p in list(df_categorical_data['cleaning_fee']) :
    if type(p) == type('abc') :
        p = p.replace(',', '' )
        price.append(float(p[1:5])) 
    else:
        price.append(np.NaN)


df_numeric_data['cleaning_fee'] = price
df_categorical_data.drop('cleaning_fee', axis = 1, inplace = True)
# =============================================================================
# 
# =============================================================================

# 3. Covertiving monthly price to float
df_categorical_data['monthly_price'].isnull().sum()


price = []
#Cleaning_fee consists of nan values that needs to be handled
for p in list(df_categorical_data['monthly_price']) :
    if type(p) == type('abc') :
        p = p.replace(',', '' )
        price.append(float(p[1:])) 
    else:
        price.append(np.NaN)
         
df_numeric_data['monthly_price'] = price
df_categorical_data.drop('monthly_price', axis = 1, inplace = True)
    

# 4. Covertiving weekly price to float
df_categorical_data['weekly_price'].isnull().sum()


price = []
#Cleaning_fee consists of nan values that needs to be handled
for p in list(df_categorical_data['weekly_price']) :
    if type(p) == type('abc') :
        p = p.replace(',', '' )
        price.append(float(p[1:])) 
    else:
        price.append(np.NaN)
         
df_numeric_data['weekly_price'] = price
df_categorical_data.drop('weekly_price', axis = 1, inplace = True)


#Scatterplot between weekly and monthly price
plt.scatter(df_numeric_data['weekly_price'], df_numeric_data['monthly_price'])

#More than 90% data is null, delete the column
df_numeric_data.drop(['monthly_price', 'weekly_price'], axis = 1, inplace = True)


#-----------------Numerical Data EDA---------------------#
#Numeric Data EDA
df_numeric_data.head()

#Deleting ID columns
df_numeric_data.drop(['id', 'scrape_id', 'host_id'], axis = 1, inplace = True)

#Check nulls
df_numeric_data.isnull().sum()

#Drop columns where null values greater than 99%
max_rows = df.shape[0]
for col in list(df_numeric_data.columns) :
    if df_numeric_data[col].isnull().sum() > (0.99 * max_rows) :
        df_numeric_data.drop(col, axis = 1, inplace = True)
 
df_numeric_data.columns

#Columns not relavant to determine house rental price
columns_to_drop = ['host_listings_count', 'host_total_listings_count',
                   'minimum_nights', 'maximum_nights',
       'minimum_minimum_nights', 'maximum_minimum_nights',
       'minimum_maximum_nights', 'maximum_maximum_nights',
       'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value',
       'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month']
df_numeric_data.drop(columns_to_drop, axis = 1, inplace= True)


#-------------------Naive Model 2------------------------
cat_columns = df_categorical_data.columns

for col in cat_columns :
    df_categorical_data[col] = df_categorical_data[col].astype('category').cat.codes
    
df_modified = pd.concat([df_numeric_data, df_categorical_data], axis = 1)

##Check for nulls and replace with 0
df_modified.isnull().sum()
df_modified.fillna(0, inplace = True)

X = df_modified.drop('price', axis = 1)
y = df_modified['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Metrics
from sklearn.metrics import mean_absolute_error, r2_score

print('RMSE : ', np.sqrt(mean_absolute_error(y_test, y_pred)))
print('R2 score : ', r2_score(y_test, y_pred))

#---------------------Model Enhancement-----------------------------

#Segregate numeric and categorical data
df_numeric_data = df._get_numeric_data()
df_categorical_data = df[list(set(df.columns) - set(df._get_numeric_data().columns))]


## Drop from numeric
columns_to_drop = ['id', 'scrape_id', 'thumbnail_url', 'medium_url', 'xl_picture_url',
       'host_id', 'host_acceptance_rate','host_listings_count', 'host_total_listings_count',
       'minimum_nights', 'maximum_nights',
       'minimum_minimum_nights', 'maximum_minimum_nights',
       'minimum_maximum_nights', 'maximum_maximum_nights',
       'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value',
       'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
       'neighbourhood_group_cleansed', 'square_feet', 'guests_included']
df_numeric_data.drop(columns_to_drop, axis = 1, inplace= True)
df_numeric_data.isnull().sum()
#drop from categories

categories_to_drop = ['host_has_profile_pic', 'neighborhood_overview', 'notes', 
                     'picture_url','first_review', 'host_since', 'summary', 
                     'interaction', 'neighbourhood', 'street', 'description',
                     'require_guest_profile_picture', 'name', 'access', 
                     'jurisdiction_names', 'host_thumbnail_url', 'transit',
                     'host_location', 'host_response_rate', 'host_url',
                     'experiences_offered', 'require_guest_phone_verification',
                     'space', 'last_review', 'host_neighbourhood', 'last_scraped',
                     'house_rules', 'host_picture_url', 'instant_bookable',
                     'requires_license', 'has_availability', 'host_name', 
                     'calendar_last_scraped', 'host_response_time', 
                     'host_is_superhost', 'host_verifications',
                     'host_about', 'listing_url', 'calendar_updated', 'license',
                     'is_business_travel_ready', 'cancellation_policy', 
                     'host_identity_verified', 'is_location_exact', 'security_deposit', 
                     'extra_people', 'cleaning_fee', 'market', 'zipcode',
                     'smart_location', 'state', 'monthly_price', 'weekly_price',
                     'city', 'country', 'country_code']
df_categorical_data.drop(categories_to_drop, axis = 1, inplace= True)
df_categorical_data.columns


################## Numeric Enhancements ########################

#Convert price to numeric
df_numeric_data['price'] = df_categorical_data['price'].apply(
        lambda p : float((p.replace(',', '' )).replace('$', '')))
df_categorical_data.drop('price', axis = 1, inplace = True)


##Treating nan values

df_numeric_data.isnull().sum()
df_categorical_data.isnull().sum()

#Categorical does not have any nan values

#check correlation for other numeric
plt.scatter(df['bathrooms'], df['bedrooms'])
plt.scatter(df['beds'], df['bedrooms'])


#drop beds as accomodates and beds highly correalted
df_numeric_data.drop('beds', axis = 1, inplace = True)

#Bedroom
df['bedrooms'].describe()

#replace null values by mode
bedroom_mode = float(df_numeric_data['bedrooms'].mode())
df_numeric_data['bedrooms'][df_numeric_data['bedrooms'].isnull()] = bedroom_mode

#treat outliers
plt.boxplot(df_numeric_data['bedrooms'][~df_numeric_data['bedrooms'].isnull()])

df_numeric_data['bedrooms'][df_numeric_data['bedrooms'] >= 5] = 5

##Bathrooms
bathroom_mode = float(df_numeric_data['bathrooms'].mode())
df_numeric_data['bathrooms'][df_numeric_data['bathrooms'].isnull()] = bathroom_mode

#price variable

plt.boxplot(df_modified['price'])

df_numeric_data['price'].describe()

#Combine 
df_modified = pd.concat([df_numeric_data, df_categorical_data], axis = 1)

#Delete extreme value
df_modified = df_modified[df_modified['price'] != float(12345)]

#check percentile
np.percentile(df_modified['price'], [10,20,30,40,50,60,70,80,90,100])

## High whiskers greater than 10 %
temp = df_modified[df_modified['price'] > 225]
temp['price'].describe()
temp['price'].hist()

df_modified = df_modified[df_modified['price'] < 1000]


##Lower whiskers less than 10th percentile
temp = df_modified[df_modified['price'] <31]
temp['price'].describe()
temp['price'].hist()

df_modified = df_modified[df_modified['price'] > 14]



#Correlation matrix
corr = df_numeric_data.corr()
#Set figure size
plt.figure(figsize=(13,8))
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


##Plot latitude longitude on map
plt.scatter(df_numeric_data['latitude'], df['longitude'], 
            c=df_numeric_data['price'], cmap = 'rainbow')


######    Categorical enhancements  ############

df_categorical_data.columns

df['market'] .isnull().sum()
df['market'].value_counts()
temp = df[df['market'] != 'London']
temp[['longitude', 'latitude']].describe()
plt.scatter(temp['latitude'], temp['longitude'])


df['neighbourhood'].isnull().sum()
df['neighbourhood'].value_counts()
plt.scatter(df['latitude'], df['longitude'], labels = df['neighbourhood'], cmap='rainbow')

#amenities count

df_modified['amenities_len'] = df_modified['amenities'].apply(lambda x : len(x.split(',')))
df_modified.drop('amenities', axis = 1, inplace = True)

df_modified.columns
df_modified['bed_type'].value_counts()
#Delete as 90%data same
df_modified.drop('bed_type', axis = 1, inplace = True)

#Property type
df_modified['property_type'].value_counts()

house_type = []
for h in list(df_modified['property_type']) :
    if h == 'Apartment' :
        house_type.append(1)
    elif h == 'House':
        house_type.append(2)
    elif h in ['Townhouse', 'Condominium', 'Serviced apartment'] :
        house_type.append(3)
    elif h in ['Loft','Bed and breakfast','Guest suite','Guesthouse',
               'Boutique hotel','Hotel','Hostel','Bungalow'] :
        house_type.append(4)
    else :
        house_type.append(5)
        
df_modified['property__type'] = house_type
df_modified.drop('property_type', axis = 1, inplace = True)

#
df_modified['room_type'].value_counts()

#Neighbourhood
df_modified['neighbourhood_cleansed'].value_counts()

#Relation
plt.scatter(df_modified['price'], df_modified['amenities_len'])

#get dummies
df_modified = pd.get_dummies(df_modified, columns=['room_type', 'neighbourhood_cleansed'], drop_first = True)
df_modified = pd.get_dummies(df_modified, columns=['property__type'], drop_first = True)
    
# Model

X = df_modified.drop(['price'], axis = 1)
y = df_modified['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Metrics
from sklearn.metrics import mean_absolute_error, r2_score

print('RMSE : ', np.sqrt(mean_absolute_error(y_test, y_pred)))
print('R2 score : ', r2_score(y_test, y_pred))

y = np.log1p(y)
y.hist()
