"""

Title of the Dataset: Breast Cancer Wisconsin (Diagnostic)

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34]. 

Resources: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

dataset = load_breast_cancer()

X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name='target')

df = pd.concat([X, y], axis=1)

# Information of the dataset
print(dataset.DESCR)

# Shape of the Dataset
print("Row of the Dataset:", df.shape[0])
print("Column of the Dataset:", df.shape[1])

# Duplicated Values
print(df.duplicated().sum())

# Checking the Dataset Contain any null values
print(df.isnull().sum())

# Describe the dataset
print(df.describe())

# Correlation between the Features and Label
print(df.corr())

# Checking the Mutual Information of the Features
def numerical_mutual_information(X, y):
  numerical_columns = X.select_dtypes(include=['number']).columns

  mi_score = mutual_info_regression(X, y)
  mi_score = pd.Series(mi_score, index=numerical_columns).sort_values(ascending=False)
  
  return mi_score

mi_score = numerical_mutual_information(X, y)
selected_col = mi_score[mi_score >= 0.1].index
print(mi_score)
print(selected_col)

# Plot the numerical features in box plot
def numerical_box_plot(df):
  numerical_column=df.select_dtypes(include=['number']).columns

  num_cols = 3
  num_rows = (len(numerical_column) + num_cols - 1) // num_cols

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
  fig.suptitle('Visualization of Box Plot', fontsize=16)

  axes = axes.flatten()

  for i, col in enumerate(numerical_column):
    sns.boxplot(x=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Box Plot of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=14)

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  
numerical_box_plot(X)

# Numerical Distribution of Features
def numerical_distribution_plot(df):
  numerical_column = df.select_dtypes(include=['number']).columns

  num_cols = 3
  num_rows = (len(numerical_column) + num_cols - 1) // num_cols

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
  fig.suptitle("Distribution of Numerical Features", fontsize=16)

  axes = axes.flatten()

  for i, col in enumerate(numerical_column):
    sns.histplot(x=df[col], ax=axes[i], palette='viridis', element='step', kde=True, stat='density')
    axes[i].set_title(f"Distribution of {col}", fontsize=14)
    axes[i].set_xlabel(col, fontsize=14)
    axes[i].set_ylabel('Density', fontsize=14)

  for j in range(i + 1, len(axes)):
    plt.delaxes(axes[j])

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  
numerical_distribution_plot(X)

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X[selected_col],
                                                     y, 
                                                     test_size=0.25, 
                                                     random_state=42, 
                                                     shuffle=False)


rfc = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

rfc.fit(X_train, y_train)

importance = rfc.feature_importances_

importance_score = pd.Series(data=importance, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(50, 6))
sns.barplot(x=importance_score.index, y=importance_score)
plt.show()

y_pred = rfc.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def feature_rebalance_dataset(X, y):
  smote = SMOTE(random_state=42)

  X, y = smote.fit_resample(X, y)

  return X, y
  
  
def feature_scaler(X, y):
  scaler = StandardScaler()

  X = scaler.fit_transform(X)

  return X, y.to_numpy()
  
# Feature Rebalance for training
X_train, y_train = feature_rebalance_dataset(X_train, y_train)

# Feature Scaling for training and testing
X_train, y_train = feature_scaler(X_train, y_train)
X_test, y_test = feature_scaler(X_test, y_test)

# Test with Logistic Regression
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

