import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/pima-indian-diabetest-database.csv', na_values={'Glucose':0, 'BloodPressure':0, 'SkinThickness':0, 'Insulin':0, 'BMI':0})

# IMPUTE MISSING VALUE
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

# pisah dulu variable X dan Y
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# implement knn impute
X_trans = imputer.fit_transform(X)

# NORMALIZATION MIN-MAX
# NORMALIZATION
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X_trans)
y_norm = y.to_numpy()

# BALANCE DATA | SMOTE
# split dataset
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.3)
oversample = SMOTE()
X_trainSmote, y_trainSmote = oversample.fit_resample(X_train, y_train)

# SAVE PREPROCESSED DATA
np.save('./data/preprocessed-data/X_train.npy', X_train)
np.save('./data/preprocessed-data/y_train.npy', y_test)

np.save('./data/preprocessed-data/X_trainSmote.npy', X_trainSmote)
np.save('./data/preprocessed-data/y_trainSmote.npy', y_trainSmote)

np.save('./data/preprocessed-data/X_test.npy', X_test)
np.save('./data/preprocessed-data/y_test.npy', y_test)