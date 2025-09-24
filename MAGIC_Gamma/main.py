"""

Title of Database: MAGIC gamma telescope data 2004

The data are MC generated (see below) to simulate registration of high energy
gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the
imaging technique. Cherenkov gamma telescope observes high energy gamma rays,
taking advantage of the radiation emitted by charged particles produced
inside the electromagnetic showers initiated by the gammas, and developing in the
atmosphere.

resources: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

"""

# import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# ML Model Training

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Setup the Datasets
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

path = "./data/magic04.data"

# Dataset doesn't contain any headers
df = pd.read_csv(path, header=None, names=cols)

# Check the first 5 rows
# print(df.head())

# Checking the prediction values
# print(df['class'].unique()) ['g', 'h']

# Converting the alphabetical value into numerical 
df['class'] = (df['class'] == 'g').astype(int)

print(df.head())

for label in cols[:-1]:
  plt.hist(df[df['class'] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
  plt.hist(df[df['class'] == 0][label], color='red', label='hedron', alpha=0.7, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()

# Training, Validation, Testing Datatest (Distribution)

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

# Scaling the dataset to get the matching size of data in each train, valid and test dataset

def scale_dataset(dataframe, OverSample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values
  
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  if OverSample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)
    
  data = np.hstack((X, np.reshape(y, (-1, 1))))
  
  return data, X, y

# Evenly Rebalance the train, valid, and test dataset

train, X_train, y_train = scale_dataset(train, OverSample=True)
valid, X_valid, y_valid = scale_dataset(valid, OverSample=False)
test, X_test, y_test = scale_dataset(test, OverSample=False)

# Train the Model

## KNN Model 

knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(X_train, y_train)

# y_pred = knn_model.predict(X_test)

# Accuracy 0.79 (79%) #

# print(classification_report(y_test, y_pred))


## Naive Bayes

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# y_pred = nb_model.predict(X_test)

# Accuracy 0.72 (72%) #

# print(classification_report(y_test, y_pred))

## Logistic Regression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# y_pred = lr_model.predict(X_test)

# Accuracy 0.79 (79%)

# print(classification_report(y_test, y_pred))

## Support Vector Machine

svc_model = SVC()
svc_model.fit(X_train, y_train)

# y_pred = svc_model.predict(X_test)

# Accuracy 0.87 (87%)

# print(classification_report(y_test, y_pred))

