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

# Import Library
import warnings
import string

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef

warnings.filterwarnings("ignore")

path = "magic04.data"
col = [
  "fLength", "fWidth", "fSize", "fConc", "fConc1",
  "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
  "class"
]

df = pd.read_csv(path, header=None, names=col)

# quick view
print("Training data shape:", df.shape)

# explore dataset
print("\nDataset Info:")
print(df.info())
print("--" * 20)

print(df.head())