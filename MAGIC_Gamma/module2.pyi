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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler