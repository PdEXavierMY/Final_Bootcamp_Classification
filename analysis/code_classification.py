import pandas as pd                                # panel data, for handling dataframes
pd.set_option('display.max_columns', None)         # show all columns of the dataframe

import numpy as np                                 # numerical python, linear algebra library

import pylab as plt                                # plotting library
import seaborn as sns                              # plotting library
sns.set(style='white')                             # seaborn style


from sklearn.linear_model import LogisticRegression            # logistic regression model   

from sklearn.preprocessing import MinMaxScaler                 # standarized
from sklearn.preprocessing import LabelEncoder                 # Para codificar nuestra variable a predecir

from sklearn.model_selection import train_test_split     # split data into train and test sets

import warnings
warnings.filterwarnings('ignore')