import pandas as pd
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap

# Load data and pre-processing
df = pd.read_csv('flights.csv', low_memory = False)
df['AIR_ROUTE'] = df['AIRLINE'] + df['FLIGHT_NUMBER'].astype(str)
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

data = df[: 1000000]
data = data.fillna(data.mean(numeric_only=True))
data_fd = data[data['CANCELLED'] == 0]
data_fd = data_fd.drop(['CANCELLED'], axis = 1)

data_fd.loc[:, 'ORIGIN_AIRPORT'] = data_fd['ORIGIN_AIRPORT'].astype('category').cat.codes
data_fd.loc[:, 'DESTINATION_AIRPORT'] = data_fd['DESTINATION_AIRPORT'].astype('category').cat.codes
data_fd.loc[:, 'AIR_ROUTE'] = data_fd['AIR_ROUTE'].astype('category').cat.codes
data_fd.loc[:, 'DATE'] = data_fd['DATE'].astype('category').cat.codes

# Add DELAY to dataframe
# NO DELAY : 0
# DELAY : 1
DELAY = np.zeros_like(data_fd['ARRIVAL_DELAY'])
DELAY[np.logical_or(data_fd['ARRIVAL_DELAY'] > 0, data_fd['DEPARTURE_DELAY'] > 0)] = 1
data_fd.loc[:, 'DELAY'] = DELAY

# # Add DELAY_TIME to dataframe
# NO DELAY : 0
# DELAY 0 - 15 MIN: 1
# DELAY 15 - 30 MIN: 2
# DELAY 30 - 45 MIN: 3
# DELAY 45 - 60 MIN: 4
# DELAY > 60 MIN: 5
DELAY_TIME = np.zeros_like(data_fd['ARRIVAL_DELAY'])
DELAY_TIME[np.logical_and(data_fd['ARRIVAL_DELAY'] <= 15, data_fd['ARRIVAL_DELAY'] > 0)] = 1
DELAY_TIME[np.logical_and(data_fd['ARRIVAL_DELAY'] <= 30, data_fd['ARRIVAL_DELAY'] > 15)] = 2
DELAY_TIME[np.logical_and(data_fd['ARRIVAL_DELAY'] <= 45, data_fd['ARRIVAL_DELAY'] > 30)] = 3
DELAY_TIME[np.logical_and(data_fd['ARRIVAL_DELAY'] <= 60, data_fd['ARRIVAL_DELAY'] > 45)] = 4
DELAY_TIME[data_fd['ARRIVAL_DELAY'] > 60] = 5
data_fd.loc[:, 'DELAY_TIME'] = DELAY_TIME

np.round(np.count_nonzero(DELAY == 1) / len(DELAY), 4)

# Manually drop unrelated features
drop_col = []
drop_col += ['TAIL_NUMBER','TAXI_OUT','WHEELS_OFF',\
             'WHEELS_ON','TAXI_IN','CANCELLATION_REASON','ARRIVAL_TIME', \
             'DIVERTED','MONTH','YEAR','DAY','DAY_OF_WEEK', \
             'AIRLINE', 'FLIGHT_NUMBER', 'ELAPSED_TIME', \
             'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY', \
             'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
data_f = data_fd.drop(list(set(drop_col)), axis = 1)

# colors = [(135/255, 206/255, 250/255), (1, 160/255, 122/255)]
# color_map = LinearSegmentedColormap.from_list('blue2salmon', colors, N=10)
color_map = cm.get_cmap('Oranges')
font = {'family' : 'DejaVu Sans', 'weight' : 'bold', 'size' : 22}
plt.rc('font', **font)

def getCorrelationMat(df):

    '''
    Get the correlation matrix between selected features
    :param: df
    :type: panda.DataFrame
    '''

    assert isinstance(df, pd.DataFrame), 'INVALID INPUT TYPE'

    data_f = df
    f = plt.figure(figsize = (12, 12))
    plt.matshow(data_f.corr(), fignum = f.number, cmap = color_map)
    plt.xticks(np.arange(data_f.shape[1]) + 0.1, data_f.columns, fontsize=10, rotation=45, color = '#FEF4E8')
    plt.yticks(np.arange(data_f.shape[1]) + 0.1, data_f.columns, fontsize=10, color = '#FEF4E8')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 12, color = '#FEF4E8')
    cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='#FEF4E8')
    plt.title('Correlation Matrix of Selected Features', fontsize = 28, fontweight='bold', y = -0.1, color = '#FEF4E8')
    plt.savefig('corr_mat.png', bbox_inches = 'tight')
    plt.show()

getCorrelationMat(data_f)

def getConfusionMat(model, model_name, X, y, classes):

    '''
    Display the confusion matrix of applying a classification model on the dataset, and print the accuracy rate
    :param: model
    :type: sklearn.tree._classes
    :param: model_name
    :type: str
    :param: X
    :type: numpy.ndarray
    :param: y
    :type: numpy.ndarray
    :param: classes
    :type: list
    '''

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) \
    and isinstance(model_name, str) and isinstance(classes, list), 'INVALID INPUT TYPE'
    assert X.shape[0] == y.shape[0] and y.shape[0] == np.size(y), 'INVALID INPUT SHAPE'
    for ele in classes: assert isinstance(ele, str), 'INVALID INPUT VALUE'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaled_features = StandardScaler().fit_transform(X_train, X_test)
    model = model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test) if len(classes) > 2 else model.predict(X_test)
    auc_score = metrics.roc_auc_score(y_test, y_pred, multi_class = 'ovr')
    print ('The accuracy rate is {} by using the {} classifier'.format(np.round(auc_score, 4), model_name))

    disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=classes,
            cmap=color_map,
            normalize='true',
            )

    class_type = 'Binary' if len(classes) == 2 else 'Multi-Class'
    title = 'Confusion Matrix for {} Classification ({})'.format(class_type, model_name)
    disp.ax_.set_title(title, color = '#FEF4E8', y = -0.2)
    plt.xlabel('Prediction Label', color = '#FEF4E8', fontweight = 'bold')
    plt.ylabel('True Label', color = '#FEF4E8', fontweight = 'bold')
    disp.ax_.tick_params(axis = 'x', colors = '#FEF4E8', labelsize = 18)
    disp.ax_.tick_params(axis = 'y', colors = '#FEF4E8', labelsize = 18)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)
    plt.show()

data_fml_bin = data_f.drop(['DELAY_TIME'], axis = 1)
data_bin = data_fml_bin.values
X, y = data_bin[:,:-1], data_bin[:,-1]
model = DecisionTreeClassifier()
classes = ['No Delay', 'Delay']
getConfusionMat(model, 'Decision Tree', X, y, classes)

data_fml_bin = data_f.drop(['DELAY_TIME'], axis = 1)
data_bin = data_fml_bin.values
X, y = data_bin[:,:-1], data_bin[:,-1]
model = RandomForestClassifier(random_state = 42)
classes = ['No Delay', 'Delay']
getConfusionMat(model, 'Random Forest', X, y, classes)

data_fml_mul = data_f.drop(['DELAY'], axis = 1)
data_mul = data_fml_mul.values
X, y = data_mul[:,:-1], data_mul[:,-1]
model = DecisionTreeClassifier(random_state = 100)
classes = ['No Delay', '0-15', '15-30', '30-45', '45-60', '>60']
getConfusionMat(model, 'Decision Tree', X, y, classes)

data_fml_mul = data_f.drop(['DELAY'], axis = 1)
data_mul = data_fml_mul.values
X, y = data_mul[:,:-1], data_mul[:,-1]
model = RandomForestClassifier(random_state = 42)
classes = ['No Delay', '0-15', '15-30', '30-45', '45-60', '>60']
getConfusionMat(model, 'Random Forest', X, y, classes)
