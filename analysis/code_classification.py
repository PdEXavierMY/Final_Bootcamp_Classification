import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import pylab as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('./data/creditcardmarketing.csv')