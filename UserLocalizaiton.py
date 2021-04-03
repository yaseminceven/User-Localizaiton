import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from numpy.core._multiarray_umath import ndarray
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import MinMaxScaler
# loading ujiIndoor dataset
dataset = pd.read_csv('trainingDataNew.csv')
val_data = pd.read_csv('validationData.csv')
#selecting specific floor and building from dataset
dataset = dataset[dataset['FLOOR'] == 1]
dataset = dataset[dataset['BUILDINGID'] == 0]
val_data = val_data[val_data['FLOOR'] == 1]
val_data = val_data[val_data['BUILDINGID'] == 0]
# selecting columns
X = dataset.iloc[:, :-7].values
long = dataset.iloc[:, -9].values
lat = dataset.iloc[:, -8].values
long_lat = list(zip(long, lat))
long_lat = np.array(long_lat)

X2 = val_data.iloc[:, :-7].values
long2 = val_data.iloc[:, -9].values
lat2 = val_data.iloc[:, -8].values
long_lat2 = list(zip(long2, lat2))
long_lat2 = np.array(long_lat2)

# changing RSS data range to btw 0,100
for i in range(0, 1356):
    for j in range(0, 519):
    	#since RSS=100 means no signal, it was assigned as 0.
        if (X[i][j] == 100):
            X[i][j] = 0
        else:
            X[i][j] = X[i][j] + 100
for i in range(0, 208):
    for j in range(0, 519):
        if (X2[i][j] == 100):
            X2[i][j] = 0
        else:
            X2[i][j] = X2[i][j] + 100


for m in range(1, 5):
    # splitting data
    X_train, X_test, long_lat_train, long_lat_test = sklearn.model_selection.train_test_split(X, long_lat,test_size=0.3)
    X_train = X
    long_lat_train = long_lat
    X_test = X2
    long_lat_test = long_lat2
    # feature scaling
    scaler = MinMaxScaler()
    X_train = X_train.astype('float64')
    X_train_scaled = scaler.fit_transform(X_train)

    # applying the model
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_scaled, long_lat_train)

    # predicting
    X_test_scaled = scaler.transform(X_test)
    long_lat_pred = model.predict(X_test_scaled)
    long_pred = long_lat_pred[:, 0]
    lat_pred = long_lat_pred[:, 1]

    # error calculation
    mse = sklearn.metrics.mean_squared_error(long_lat_test, long_lat_pred)
    best = 0
    r = sklearn.metrics.r2_score(long_lat_test, long_lat_pred)
    if r > best:
        best = r


print('')
print('The MSE value is:', mse)
print('The R2 score is:', best)
euclidean = sklearn.metrics.pairwise.paired_distances(
    long_lat_test, long_lat_pred)
print('The error distance:',euclidean.mean())

# plotting the data
long_test = long_lat_test[:, 0]
lat_test = long_lat_test[:, 1]

plt.scatter(long_test, lat_test, color='black')
plt.scatter(long_pred, lat_pred, color='blue')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()
