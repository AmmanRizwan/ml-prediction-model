# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pylot as plt
import seaborn as sns

# Dataset Proprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Feature Engineering
from imblearn.over_sampling import RandomOverSampler

# Dataset 

df = pd.read_csv("./data/Titanic-Dataset.csv")

# Remove the Unnecessary Columns
df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"], axis=1, inplace=True)

print(df.head())

# MetaData

print("Metadata of the Dataset:")
print("--" * 20)
print(df.info())

print("Shape of the Dataset:", df.shape)
print("Size of the Dataset:", df.size)

print("Number of Missing Values in the Dataset:")
print("--" * 20)
print(df.isnull().sum())

sns.heatmap(df.describe(), annot=True, cmap="Blues")

def visualize_box_plots(df):
  numerical_columns = df.select_dtypes(include=['number']).columns
  
  num_cols = 3
  num_rows = (len(numerical_columns) + num_cols - 1) // num_cols
  
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
  fig.suptitle("Box Plot of Numerical Features", figsize=16)
  
  axes = axes.flatten()
  
  for i, col in enumerate(numerical_columns):
    sns.boxplot(x=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Box Plot of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    
  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  
  plt.show()

# Visualize box Plots
# visualize_box_plots(df) # show an graph plot

def calculate_outliers_percentage(df):
  outlier_counts = {}
  
  for column in df.select_dtypes(include=['number']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q1 - Q3
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_counts[column] = len(outliers)
    
  for column in outlier_counts:
    percentage = (outlier_counts[column] / len(df)) * 100
    print(f'Percentage of Outliers in {column}: {percentage:.2f}%')


# Calculate the Percentage of Outlier
# calculate_outliers_percentage(df) # show the percentage

def handle_outliers(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  
  IQR = Q3 - Q1
  
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  
  df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
  
  return df

# Age has the outiler from the given graph

new_data = handle_outliers(df, 'Age')

df['Age'] = new_data["Age"].copy()

# Checking the Age Box Plot that is fixed
# visualize_box_plots(df) # show the Graph of Box Plot

# Checking Skewness

def visualize_numerical_distribution(df):
  numerical_columns = df.select_dtypes(include=['number']).columns
  
  num_cols = 3
  num_rows = (len(numerical_columns) + num_cols - 1) // num_cols
  
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
  fig.suptitle("Distribution of Numerical Features", fontsize=16)
  
  axes = axes.flatten()
  
  for i, col in enumerate(numerical_columns):
    sns.heatmap(df[col], kde=True, ax=axes[i], palette='viridis', element='step', stat='density')
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)
    
  
  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  
  

    
    

