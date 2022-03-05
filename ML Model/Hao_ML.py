import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('flights.csv', low_memory = False)

# Combine AIRLINE AND FLIGHT_NUMBER to a string, call it AIR_ROUTE
df['AIR_ROUTE'] = df['AIRLINE'] + df['FLIGHT_NUMBER'].astype(str)

# Convert YEAR, MONTH, DAY, DAY_OF_WEEK to datetime type, call it DATE
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Print out name of features/columns
print ('There are {} features available in this dataset, which are: '.format(df.shape[1]))

for col in df.columns: print (col)

# Fill nan in columns
data = df[:]
data = data[0 : 1000000]
data = data.fillna(data.mean(numeric_only=True))

# Filter data based on cancellation status
data_f = data[data['CANCELLED'] == 0]
data_f = data_f.drop(['CANCELLED'], axis = 1)

# Convert object type to catergorical type in pandas
data_f.loc[:, 'ORIGIN_AIRPORT'] = data_f['ORIGIN_AIRPORT'].astype('category').cat.codes
data_f.loc[:, 'DESTINATION_AIRPORT'] = data_f['DESTINATION_AIRPORT'].astype('category').cat.codes
data_f.loc[:, 'AIR_ROUTE'] = data_f['AIR_ROUTE'].astype('category').cat.codes
data_f.loc[:, 'DATE'] = data_f['DATE'].astype('category').cat.codes

# Add DELAY to dataframe
DELAY = np.zeros_like(data_f['ARRIVAL_DELAY'])
DELAY[data_f['ARRIVAL_DELAY'] > 0] = 1
data_f.loc[:, 'DELAY'] = DELAY

data_f.corr()

data_f.info()

# Find columns with insufficiant valuable info
# for col in df.columns:
#     null_sum = df[col].isnull().sum()
#     null_per = null_sum * 100 / df.shape[0]
#     if null_per == 0: continue
#     # print ('{} column has {} percentage of nan values'.format(col, null_per))
#     if null_per > 80: drop_col.append(col)

# print (drop_col)

drop_col = []

# Manually select unrelevant columns
drop_col += ['DISTANCE','TAIL_NUMBER','TAXI_OUT','SCHEDULED_TIME','WHEELS_OFF',\
             'WHEELS_ON','TAXI_IN','CANCELLATION_REASON','ARRIVAL_TIME', \
             'DIVERTED','MONTH','YEAR','DAY','DAY_OF_WEEK', \
             'AIRLINE', 'FLIGHT_NUMBER']

data_f = data_f.drop(list(set(drop_col)), axis = 1)

data_f.info()

# drop_col += ['DISTANCE','TAIL_NUMBER','TAXI_OUT','SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF',\
#              'ELAPSED_TIME','AIR_TIME','WHEELS_ON','TAXI_IN','CANCELLATION_REASON','ARRIVAL_TIME', \
#              'SCHEDULED_ARRIVAL','DIVERTED','MONTH','YEAR','DAY','DAY_OF_WEEK', \
#              'AIRLINE', 'FLIGHT_NUMBER']

f = plt.figure(figsize = (15, 15))
plt.matshow(data_f.corr(), fignum = f.number)
plt.xticks(np.arange(data_f.shape[1]) + 0.25, data_f.columns, fontsize=10, rotation=45)
plt.yticks(np.arange(data_f.shape[1]) + 0.1, data_f.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 12)
plt.title('Correlation Matrix of Selected Features', fontsize = 12, y = -0.1)
plt.show()

# Get the data for building the model
# data_fml = data_f[['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', \
#                    'SCHEDULED_ARRIVAL', 'DATE', 'DELAY']]

# data_fml = data_f[['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', \
#                    'SCHEDULED_ARRIVAL', 'DATE', 'AIR_ROUTE', 'DELAY']]

data_fml = data_f.drop(['ARRIVAL_DELAY'], axis = 1)

data_fml.info()

data_ml = data_fml.values
X, y = data_ml[:,:-1], data_ml[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Decision Tree
scaled_features = StandardScaler().fit_transform(X_train, X_test)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

# Predict and get the prediction score
pred_prob = clf.predict_proba(X_test)
auc_score = metrics.roc_auc_score(y_test, pred_prob[:,1])
print ('The accuracy rate is {} by using the decision tree classifier'.format(np.round(auc_score, 4)))

# Random Forest
rf = RandomForestRegressor(random_state = 42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
auc_score = metrics.roc_auc_score(y_test, y_pred)
print ('The accuracy rate is {} by using the random forest model'.format(np.round(auc_score, 4)))

print ('The probability of delay for a random flight is {}'.format(np.round(np.count_nonzero(DELAY == 0) / len(DELAY), 4)))
