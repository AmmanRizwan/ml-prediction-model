"""
Title of the Dataset: Titanic Dataset

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable”
RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough 
lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.

Resources: https://www.kaggle.com/datasets/yasserh/titanic-dataset

"""

# Import Libaraies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import boxcox
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

train = pd.read_csv("./data/Titanic-Dataset.csv")
train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X, y = train.drop(columns=['Survived'], axis=1), train['Survived']

# Summary

## Information of the dataset
print(train.info())

"""
- Dataset contain Features and Label
- No Duplicates
- Some Nullable Values in both categorical and numerical features
"""

## Shape of the Dataset

print("Row of the Dataset:", train.shape[0])
print("Column of the Dataset:", train.shape[1])

## Check the null values in each columns

print(train.isnull().sum())

## Describing the dataset

print(train.describe())

## Find the correlation between the features and label

def correlation_matrix(df):
  numerical_column = df.select_dtypes(include=['number']).columns

  corr = df[numerical_column].corr()['Survived'].sort_values(ascending=False)
  corr_abs = corr.abs().sort_values(ascending=False)
  print(corr_abs)

correlation_matrix(train)

# Visualize

## Box Plot for the Numerical Columns

def visualize_box_plot(df):
  numerical_column = df.select_dtypes(include=['number']).columns

  num_cols = 3
  num_rows = (len(numerical_column) + num_cols - 1) // num_cols

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
  fig.suptitle("Box Plot of the Numerical Distribution", fontsize=16)

  axes = axes.flatten()

  for i, col in enumerate(numerical_column):
    sns.boxplot(x=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f"Box Plot of {col}", fontsize=14)
    axes[i].set_xlabel(col, fontsize=14)

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  
visualize_box_plot(train)

## Distribution for the Numerical Columns

def visualize_numerical_distribution(df):
  numerical_column = df.select_dtypes(include=['number']).columns

  num_cols = 3
  num_rows = (len(numerical_column) + num_cols - 1) // num_cols

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
  fig.suptitle("Distribution of Numerical Features", fontsize=16)

  axes = axes.flatten()

  for i, col in enumerate(numerical_column):
    sns.histplot(df[col], kde=True, ax=axes[i], palette='viridis', element='step', stat='density')
    axes[i].set_title(f"Distribution of {col}", fontsize=14)
    axes[i].set_xlabel(col, fontsize=14)
    axes[i].set_ylabel('Density', fontsize=14)

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  
visualize_numerical_distribution(train)

## Distribution for the Categorical Columns

def visualize_categorical_distribution(df):
  categorical_columns = df.select_dtypes(include=['object']).columns

  num_cols = 3
  num_rows = (len(categorical_columns) + num_cols - 1) // num_cols

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
  fig.suptitle("Distribution of Categorical Feature", fontsize=16)

  axes = axes.flatten()

  for i, col in enumerate(categorical_columns):
    sns.countplot(data=df, x=col, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Distribution of {col}', fontsize=16)
    axes[i].set_xlabel(col, fontsize=14)
    axes[i].set_ylabel('Count', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45)

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  
visualize_categorical_distribution(train)


def visualization_correlation(df):
  numerical_columns = df.select_dtypes(include=['number']).columns
  
  corr = df[numerical_columns].corr()
  
  plt.figure(figsize=(10, 6))
  sns.heatmap(corr, annot=True, fmt='.2f', cmap="Blues")
  plt.title("Correlation of Numerical Distribution")
  plt.show()
  
visualization_correlation(train)

# Feature Engineering

## Handling the Missing Values

X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
X['Age'] = X['Age'].interpolate(method='spline', order=2).round()

## Checking the mutual information of the features
def categorical_mutual_information(X, y):
  categorical_columns = X.select_dtypes(include=['object']).columns

  order_encoded = OrdinalEncoder()
  categorical_encoded = order_encoded.fit_transform(X[categorical_columns])
  mi_score = mutual_info_classif(X=categorical_encoded,
                                 y=y,
                                 discrete_features=True,
                                 random_state=42)

  mi_score = pd.Series(mi_score, index=categorical_columns).sort_values(ascending=False)
  print(mi_score)

def numerical_mutual_information(X, y):
  numerical_columns = X.select_dtypes(include=['number']).columns

  mi_score = mutual_info_regression(X[numerical_columns],
                                    y,
                                    random_state=42)
  mi_score = pd.Series(mi_score, index=numerical_columns)

  print(mi_score)
  
categorical_mutual_information(X, y)

numerical_mutual_information(X, y)

## Converting the Categorical into Numerical Order

## One Hot Encoding Feature

def feature_one_hot_encoding(df, column, drop_column=False):
  hot_encoder = OneHotEncoder(sparse_output=False, drop=None)
  matrix = hot_encoder.fit_transform(df[[column]])
  ohe_cols = sorted(df[column].unique())

  ohe_df = pd.DataFrame(matrix, columns=ohe_cols)
  df = pd.concat([df, ohe_df], axis=1)

  if drop_column:
    df.drop(columns=[column], axis=1, inplace=True)

  return df

## Label Encoding Feature

def feature_label_encoding(df, column):
  label_encoding = LabelEncoder()

  df[column] = label_encoding.fit_transform(df[column])

  return df[column]

X = feature_one_hot_encoding(X, 'Embarked', drop_column=True)

X['Sex'] = feature_label_encoding(X, 'Sex')

## Calculating the Outliers Percentage

def calculate_outliers_percentage(df):
  outlier_counts = {}

  for column in df.select_dtypes(include=['number']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_counts[column] = len(outliers)

  for column in outlier_counts:
    percentage = (outlier_counts[column] / len(df)) * 100
    print(f'Percentage of Outliers in {column}: {percentage:.2f}%')
    
## Handling the Outlier from the features
    
def handle_outliers(df):
  for column in df.select_dtypes(include=['number']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
  
  return df

## Handling the Skewness from the features

def handle_skewness(df, threshold=1.0):
  numerical_columns = df.select_dtypes(include=['number']).columns
  lambda_dict = {}

  for col in numerical_columns:
    skewness = df[col].skew()
    if skewness > threshold:
      df[col] = df[col] + 1
      df[col], fitted_lambda = boxcox(df[col])
      lambda_dict[col] = fitted_lambda

  return df, lambda_dict

# Handling the outlier to check the prediction score

X, lambda_dict = handle_skewness(X)

# Spliting the training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)

## Random Forest Classifier Prediction

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 

# Importance of Each Features

importance = rfc.feature_importances_

importance_score = pd.Series(importance, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance_score.index, y=importance_score)
plt.title("Importance of Each Features")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

## XGBoost Classifier Prediction

xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Non Traditional Model

"""
- Let's Balance the DataFrame of the categorical and numerical dataset

- Convert the dataframe into a scaler for model fitting
"""

## Balance the dataset

def feature_rebalance_data(X, y):
  smote = SMOTE(random_state=42)

  X, y = smote.fit_resample(X, y)

  return X, y

## Convert the dataset into scaler

def feature_scaling(X, y):
  scaler = StandardScaler()

  X = scaler.fit_transform(X)

  return X, y.to_numpy()


"""
Rebalance:

- Training Set Need to rebalance the dataset

Scaler:
- Training and Testing Set Need to Scaler to fit on the dataset
"""

## Rebalance
X_train, y_train = feature_rebalance_data(X_train, y_train)

## Scaler
X_train, y_train = feature_scaling(X_train, y_train)
X_test, y_test = feature_scaling(X_test, y_test)

## Logistic Regression

lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

## KNeighbors Classifier

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

## Naives Bayes

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

## Support Vector Machine

svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

## Nerual Network Model

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(9, )),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_pred = nn_model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
Random Forest Classifier
- Accuracy: 85%
- Precision:
  - 1: 82%
  - 0: 87%

XGBoost Classifier
- Accuracy: 83%
- Precision:
  - 1: 77%
  - 0: 85%

LogisticRegression
- Accuracy: 78%
- Precision:
  - 1: 66%
  - 0: 88%
  
KNeighbors Classifier
- Accuracy: 81%
- Precision:
  - 1: 81%
  - 0: 81%

GaussianNB
- Accuracy: 80%
- Precision:
  - 1: 88%
  - 0: 70%
  
SVC
- Accuracy: 84%
- Precision:
  - 1: 82%
  - 0: 85%

NN
- Accuracy: 85%
- Precision:
  - 1: 76%
  - 0: 91%
"""