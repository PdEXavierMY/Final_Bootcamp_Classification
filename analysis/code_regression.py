import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore

warnings.filterwarnings('ignore')

print(Fore.CYAN +"Cargamos el csv con la librería pandas"); print(Fore.RESET)
dataframe = pd.read_csv('data/regression_data.csv', delimiter=';')
print(dataframe.head())

print(Fore.CYAN + "Examinamos el csv") ; print(Fore.RESET)
print('Las columnas son:\n')
print(dataframe.columns)
print('La descripción del csv es:\n')
print(dataframe.describe())

print(Fore.CYAN + """Observaciones:
Las columnas no numéricas son : date, bathrooms, floor, lat, long
Como queremos predecir el precio, la columna que va a ir por separado es price
Para predecir el precio necesitamos: bedrooms, bathrooms, sqft, floors, condiciones, grade, year_built, year_renovate, waterview
lat, long y zipcode no nos influye ya que estan todas las casas en la misma zona de Seattle, Tacoma y alrededores, gracias a Google Maps""") ; print(Fore.RESET)


print(Fore.CYAN + "Pasamos a la limpieza de los datos") ; print(Fore.RESET)

print(Fore.CYAN + "Hacemos una copia de los datos") ; print(Fore.RESET)

copia_dataframe = dataframe.copy()

print(Fore.CYAN + "Los datos que son floats que no tengan sentido, como 2,5 baños, vamos a pasarlos a integers:") ; print(Fore.RESET)


def redondear_datos(columna):
    for i in range(len(dataframe)):
        if ',' in dataframe[columna][i]:
            dataframe[columna][i] = float(dataframe[columna][i].replace(',', '.'))
            dataframe[columna][i] = round(dataframe[columna][i])
    return dataframe


datos_a_redondear = ['bathrooms', 'floors']

for i in datos_a_redondear:
    redondear_datos(i)
    print(dataframe[i])

print(Fore.CYAN + "Ahora vamos a quitar las columnas que no nos hacen falta: lat, long, date, zipcode") ; print(Fore.RESET)

dataframe = dataframe.drop(columns=['date', 'lat', 'long', 'zipcode'], axis=1)

print(Fore.CYAN + """Para facilitar el análisis, vamos a agrupar todas las columnas que contengan mediciones en metros cuadrados, creando así una única columna con todos los metros cuadrados de la casa.
Tendremos al final dos columnas: la medida del terreno (lot) y la medida de la casa (living)
No nos interesan las medidas anteriores a las reformas ya que no existen.
above y basement los quitamos ya que living es la suma de estas, y en el precio influyen los metos cuadrados totales""") ; print(Fore.RESET)

dataframe = dataframe.drop(columns=['sqft_above', 'sqft_basement',
             'sqft_living', 'sqft_lot'], axis=1)
print(dataframe.head())
print(dataframe.columns)

print(Fore.CYAN + """Como hay muy pocas casas renovadas, hay muchos ceros en la columna renovate.
Por tanto, podemos eliminar esa columna adaptando el año de construcción con el año de renovacion de las casas que se han renovado""") ; print(Fore.RESET)

for i in range(len(dataframe['yr_renovated'])):
    if dataframe['yr_renovated'][i] != 0:
        dataframe['yr_built'][i] = dataframe['yr_renovated'][i]

dataframe = dataframe.drop(columns=['yr_renovated'], axis=1)
print(dataframe['yr_built'])

print(Fore.CYAN + """Las columnas que tengan que ver con numero de habitaciones, las podemos agrupar para simpllificar los calculos
Creamos una nueva columna llamada habitaciones que será la suma de las columnas bedrooms y bathrooms""" ); print(Fore.RESET)

dataframe['habitaciones'] = 0
for i in range(len(dataframe['bedrooms'])):
    dataframe['habitaciones'][i] = int(dataframe['bedrooms'][i]) + int(dataframe['bathrooms'][i])

print(Fore.CYAN + "Exportamos el csv organizado y limpio") ; print(Fore.RESET)

dataframe.to_csv('data/regression_data_clean.csv', index=False)

print(Fore.CYAN + "Una vez hecha la limpieza, vamos a pasar a hacer la regresión") ; print(Fore.RESET)

print(Fore.CYAN + "Cargamos el csv limpio") ; print(Fore.RESET)

dataframe_regresion = pd.read_csv('data/regression_data_clean.csv', delimiter=',')

print(Fore.GREEN + "How many rows of data do you have?")

print(dataframe_regresion.shape)

print(Fore.GREEN + """4.  Find the unique values of the following columns:
    - What are the unique values in the column `bedrooms`?
    - What are the unique values in the column `bathrooms`?
    - What are the unique values in the column `floors`?
    - What are the unique values in the column `condition`?
    - What are the unique values in the column `grade`?
""") ; print(Fore.RESET)

valores_unicos = ['bedrooms', 'bathrooms', 'floors', 'condition', 'grade']

for i in valores_unicos:
    print(dataframe_regresion[i].unique())