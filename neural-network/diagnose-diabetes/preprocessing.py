import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('./data/diabetes.csv', na_values={
                 'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 0})

# IMPUTE MISSING VALUE
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

# pisah dulu variable X dan Y
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

X_trans = imputer.fit_transform(X)

# print(sum(np.isnan(X_trans).flatten()))

# NORMALIZATION
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X_trans)
y_norm = y.to_numpy()

# print(X_norm)
# print(y_norm)

# SMOTE
oversample = SMOTE()

# split data train and test

# X_smote, y_smote = oversample.fit_resample(X_norm, y_norm)
# unique, count = np.unique(y_smote, return_counts=True)

# y_trainSmoted_value_count = {k: v for (k, v) in zip(unique, count)}

# print(y_trainSmoted_value_count)

# print(X_smote)
# print(y_smote)

# impute dulu
# normalization
# smote
# pre-trained model
