import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import pylab as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression            # logistic regression model   
from sklearn.preprocessing import MinMaxScaler                 # standarized
from sklearn.preprocessing import LabelEncoder                 # Para codificar nuestra variable a predecir

from sklearn.model_selection import train_test_split     # split data into train and test sets

import warnings
warnings.filterwarnings('ignore')