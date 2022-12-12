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
print(data) #vemos que se ha importado correctamente
print(data.head(10), "\n") #vemos las 10 primeras filas del dataset

#sacamso la información de los datos para revisar si necesitamos hacer algún ajuste a nuestros dataset
#analizamos el dataset

print("Estructura de los datos: ", data.shape, "\n") #vemos las filas y columnas que tenemos
print(data.describe(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")
#Inside de lo printeado

#Hay varias columnas con varos nulos que analizaremos más adelante, y varias columnas no numericas
#que también modificaremos.

#primero miraremos la correlación de las variables numéricas

lista = data.select_dtypes(include=['int64', 'float64']).columns; print(lista)
corr = data[lista].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True)
plt.show()

#Vemos que no hay ninguna variable que esté muy correlacionada con otra, por lo que no es necesario eliminar ninguna columna basandose en este criterio.
