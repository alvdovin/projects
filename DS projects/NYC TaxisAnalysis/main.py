
# For additional information and documentation please, see attached file: NYCTaxisAnalysisDocumentation
# along with this script.

# Import all necessary libraries:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
import scipy.stats as stats

# Question 1:
#1.a: Programmatically download and load into your favorite analytical tool the trip data for September 2015.
# use pandas library to read csv file:
df = pd.read_csv('https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv');

#1.b: Report how many rows and columns of data you have loaded:
# To get number of rows and columns use shape function from pandas:
print("Number of rows: "+str(df.shape[0])+", Number of columns: "+str(df.shape[1]));
# Store original number of records:
orig_num_records=df.shape[0]

# Question 2:
#2.a: Plot a histogram of the number of the trip distance ("Trip Distance"):
# Define basic properties for histogram and legend:
plt.title('Trip Distance Histogram')
plt.xlabel('Trip Distance')
plt.ylabel('Number Of Observations')
# Distribution of Trip Distance variable is positive skewed distribution
# Therefore limitation on x-axis gives petter histogram visualization results
bins = np.linspace(0, 15, 200)
plt.hist(df['Trip_distance'], bins, alpha=0.7)
plt.show()

#2.b: Report any structure you find and any hypotheses you have about that structure.
df['Trip_distance'].describe()


# Question 3:
# 3.a: Report mean and median trip distance grouped by hour of day.

# Use datetime function from datetime module and convert to datetime format:
df['pickup_datetime'] = df['lpep_pickup_datetime'].apply(lambda x: 
                                                       datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: x.hour)

df['dropoff_datetime'] = df['Lpep_dropoff_datetime'].apply(lambda x: 
                                                       datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
# Define field: time_travelled. We will use it for future analysis:
# Store trip time in different units:
df['trip_time_seconds'] =(df['dropoff_datetime'] - df['pickup_datetime']).apply(lambda x: x.total_seconds())
df['trip_time_minutes']=df['trip_time_seconds']/60
df['trip_time_hours']=df['trip_time_minutes']/60
 
# Mean Trip Distance Plot:
mean=df[['Trip_distance','pickup_hour']].groupby('pickup_hour').mean()
mean.plot.bar()
plt.title('Mean Trip Distance By Hour Of Day')
plt.xlabel('Pickup Hour')
plt.legend(["Mean Trip Distance"])
plt.show()

# Median Trip Distance Plot:
df[['Trip_distance','pickup_hour']].groupby('pickup_hour').median().plot.bar(label = 'Mean Trip Distance')
plt.title('Median Trip Distance By Hour Of Day')
plt.xlabel('Pickup Hour')
plt.legend(["Median Trip Distance"])
plt.show()

# 3.b: We'd like to get a rough sense of identifying trips that originate or terminate at one of the NYC area airports.
# Can you provide a count of how many transactions fit this criteria, the average fair, and any other interesting characteristics of these trips.   
    
# To find distance between two coordinates, let's define haversine function:
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

# is_NYC_airport_trip function identifies if trip was originated or terminated in one of NYC airports:
def is_NYC_airport_trip(v):    
    # Check if Trip was originated or terminated at JFK:
        if ((haversine(-73.778743,40.639826,v['Pickup_longitude'],v['Pickup_latitude']) < 2.52) | 
        (haversine(-73.778743,40.639826,v['Dropoff_longitude'],v['Dropoff_latitude']) < 2.52) |

         # Check if Trip was originated or terminated at LaGuardia:
        (haversine(-73.875136,40.777324,v['Pickup_longitude'],v['Pickup_latitude']) < 2.52) |
        (haversine(-73.875136,40.777324,v['Dropoff_longitude'],v['Dropoff_latitude']) < 2.52)):
            return True
        else:
            return False

# Compute array to store information if trip was originated\terminated at one of the airports:
is_airport_trip = [None]*df.shape[0]
for it, v in df.iterrows():
        is_airport_trip[it] = is_NYC_airport_trip(v)

# Add variable to identify airport trip:     
df['is_airport_trip'] = is_airport_trip

# Print results on how many trips were originated\terminated at one of the airports:
print('Out of '+str(len(df.index))+' trips, '  +str(len(df[df.is_airport_trip==True].index))+' originated or terminated in one of the airports')

# Average fair of airport trips:
print('Average fare of airport trips is '+str(df[df['is_airport_trip']== True]['Fare_amount'].mean()))
# Average fair of non-airport trips:
print('Average fare of non-airport trips is '+str(df[df['is_airport_trip']== False]['Fare_amount'].mean()))

# Average trip distance of airport trips:
print('Average trip distaince of airport trips is '+str(df[df['is_airport_trip']== True]['Trip_distance'].mean()))
# Average trip distance of non-airport trips:
print('Average trip distaince of non-airport trips is '+str(df[df['is_airport_trip']== False]['Trip_distance'].mean()))

# Average tip amount of airport trips:
print('Average tip amount of airport trips is '+str(df[df['is_airport_trip']== True]['Tip_amount'].mean()))
# Average tip amount of non-airport trips:
print('Average tip amount of non-airport trips is '+str(df[df['is_airport_trip']== False]['Tip_amount'].mean()))

# Average number of passangers of airport trips:
print('Average fare of airport trips is '+str(df[df['is_airport_trip']== True]['Passenger_count'].mean()))
# Average number of passangers of non-airport trips:
print('Average fare of non-airport trips is '+str(df[df['is_airport_trip']== False]['Passenger_count'].mean()))


# Histogram which showing number of airport trips by pickup hour: 
plt.title('Number of Airport Trips by Hour')
plt.xlabel('Pickup Hour')
plt.ylabel('Number Of Observations')
# Distribution of Trip Distance variable is positive skewed distribution
# Therefore limitation on x-axis gives petter histogram visualization results
bins = np.linspace(0, 24, 50)
plt.hist(df[df.is_airport_trip==True]['pickup_hour'], bins, alpha=0.7)
plt.show()


# Question 4
# 4.a: Build a derived variable for tip as a percentage of the total fares
df['tip_percentage'] = df['Tip_amount']/df['Fare_amount']

# 4.b: Build a predictive model for tip as a percentage of the total fare

# Exclude outliers and other bogus records:
# To determine what records to exclude, 
# let's look at the plot of Tip_amount vs Fare_amount: 
plt.plot(df['Tip_amount'],df['Fare_amount'],'ro')
plt.xlabel('Tip Amount')
plt.ylabel('Fare Amount')

# Exclude records where Tip Amount and Tolls Amount are less than or equal to '0':
df=df[df['Tip_amount']>=0]
df=df[df['Tolls_amount']>=0]
# The initial fare is $2.5 which we assume would the least amount charged
df=df[df['Fare_amount']>2.5] 
# Exclude records where trip time is '0':
df=df[df['trip_time_seconds']>0]
# Exclude records where trip distance is '0':
df=df[df['Trip_distance']>0]
# Exclude records where Tip Amount is greater than Fare Amount:
df=df[df['tip_percentage']<1]
# Exclude obvious outliers:
df=df[df['Trip_distance']<200]
# Print number of original vs cleaned number of records:
print('Number of records after cleaning is: '+str(df.shape[0])+'. This is: : '+str(round(100*(df.shape[0]/orig_num_records),2))+'% of original values.')

# We will build prediction model using  sklearn library which is working on numpy array.
# Therefore we need to convert all relevant columns in existing dataframe df to numpy arrays:

# Let's define matrix of features X:
# We will transform and convert all numerical columns first:
# To combine a few attributes, we use numpy operation c_ and as_matrix() to convert attributes into numpy arrays:
X = np.c_[df['Passenger_count'].as_matrix(),df['Trip_distance'].as_matrix(),df['Fare_amount'].as_matrix(),
         df['Extra'].as_matrix(),df['MTA_tax'].as_matrix(),df['Tolls_amount'].as_matrix(),
         df['improvement_surcharge'].as_matrix()]
    
# Now, let's define and load all categorical features.
# To work with categorical attributes, we will use one hot encoding method to handle all such type of attributes as numeric:
for atr in ['VendorID', 'Store_and_fwd_flag', 'Payment_type','Trip_type ','pickup_hour','is_airport_trip']:
    X_orig = df[atr].as_matrix()
    le = preprocessing.LabelEncoder() #define object to perform encoding
    le.fit(X_orig.reshape(X_orig.shape[0],1)) # Encode labels between 0 and n_classes-1
    ans = le.transform(X_orig) # Transform labels to normalized encoding

    enc = OneHotEncoder()
    enc.fit(ans.reshape(X_orig.shape[0],1))
    ans2 = enc.transform(ans.reshape(X_orig.shape[0],1)).toarray()
    X = np.c_[X,ans2]

# Define dependent variable:
y = df['tip_percentage'].as_matrix()

# Build Regression Model:
# Create linear regression object:
regr = linear_model.LinearRegression()
# Train the model using the training set:
regr.fit(X, y)
# Print coefficients:
print('Coefficients: \n', regr.coef_)
# The mean squared error:
print("Mean squared error: %.2f" % np.mean((regr.predict(X) - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X, y))

# Question 5 A (Distributions)
#5.a: Build a derived variable representing the average speed over the course of a trip
df['average_speed'] = df['Trip_distance']/df['trip_time_hours']

#5.b: Can you perform a test to determine if the average trip speeds are materially the same in all weeks of September? 
# If you decide they are not the same, can you form a hypothesis regarding why they differ?
# Define 'week' variable to further unumerate each trip with it's respective week of the month:
df['week_of_the_year'] = df['pickup_datetime'].apply(lambda x: x.week)
dif_val=df['week_of_the_year'].min()-1
df['week_of_the_month']=df['week_of_the_year']-dif_val

# Perform ANOVA test to determine if the average trip speeds among weeks are stastistically different:
    
# Load average speed attribute into different array split by week:
first_week = df['average_speed'][df['week_of_the_month']==1].as_matrix()
second_week = df['average_speed'][df['week_of_the_month']==2].as_matrix()
third_week= df['average_speed'][df['week_of_the_month']==3].as_matrix()
fourth_week = df['average_speed'][df['week_of_the_month']==4].as_matrix()
fifth_week = df['average_speed'][df['week_of_the_month']==5].as_matrix()

# Perform One Way ANOVA:
stats.f_oneway(first_week,second_week, third_week,fourth_week, fifth_week)

#5.c: Can you build up a hypothesis of average trip speed as a function of time of day?

# Again let's build ANOVA hypothesis for the hour of the day:
# Group data-set by the hour of the day:
groupby_hour = df.groupby('pickup_hour')
hour_of_the_day = []

for name,df in groupby_hour:
    hour_of_the_day.append(df['average_speed'])
    
hr = hour_of_the_day

# Build ANOVA to perform test for hours of the day:
stats.f_oneway(hr[0],hr[1],hr[2],hr[3], hr[4],hr[5],
               hr[6],hr[7],hr[8],hr[9],hr[10],hr[11],
               hr[12],hr[13],hr[14],hr[15],hr[16],hr[17],
               hr[18],hr[19],hr[20],hr[21],hr[22],hr[23])
