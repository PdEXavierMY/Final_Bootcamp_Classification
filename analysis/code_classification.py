#Vamos a realizar un modelo de classification en los datos de unos clientes de banco 
# para estudiar y predecir el comportamiento de los clientes en función de sus características

#primero importamos las librerías necesarias
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import pylab as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Importamos los datos
data = pd.read_csv('./data/creditcardmarketing.csv')

#sacamso la información de los datos para revisar si necesitamos hacer algún ajuste a nuestros dataset
print(data.head(), "\n")
print("Estructura de los datos: ", data.shape, "\n")
print(data.describe(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")