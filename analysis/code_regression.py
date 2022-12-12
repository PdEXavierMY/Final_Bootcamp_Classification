import pandas as pd
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore

warnings.filterwarnings('ignore')

print("Cargamos el csv con la librería pandas")

dataframe = pd.read_csv('data/regression_data.csv', delimiter=';')
print(dataframe.head())

print("Examinamos el csv")

print('Las columnas son:\n')
print(dataframe.columns)
print('La descripción del csv es:\n')
print(dataframe.describe())

print("""Observaciones:
Las columnas no numéricas son : date, bathrooms, floor, lat, long
Como queremos predecir el precio, la columna que va a ir por separado es price
Para predecir el precio necesitamos: bedrooms, bathrooms, sqft, floors, condiciones, grade, year_built, year_renovate, waterview
lat, long y zipcode no nos influye ya que estan todas las casas en la misma zona de Seattle, Tacoma y alrededores, gracias a Google Maps""")

print("Pasamos a la limpieza de los datos")


print("Hacemos una copia de los datos")


copia_dataframe = dataframe.copy()

print("Los datos que son floats que no tengan sentido, como 2,5 baños, vamos a pasarlos a integers:")



def redondear_datos(columna):
    for i in range(len(dataframe)):
        if ',' in dataframe[columna][i]:
            dataframe[columna][i] = float(
                dataframe[columna][i].replace(',', '.'))
            dataframe[columna][i] = round(dataframe[columna][i])
        else:
            dataframe[columna][i] = int(dataframe[columna][i])
    return dataframe


datos_a_redondear = ['bathrooms', 'floors']

for i in datos_a_redondear:
    redondear_datos(i)
    print(dataframe[i])


print("Ahora vamos a quitar las columnas que no nos hacen falta: lat, long, date, zipcode")


dataframe = dataframe.drop(columns=['date', 'lat', 'long'], axis=1)

print("""Para facilitar el análisis, vamos a agrupar todas las columnas que contengan mediciones en metros cuadrados, creando así una única columna con todos los metros cuadrados de la casa.
Tendremos al final dos columnas: la medida del terreno (lot) y la medida de la casa (living)
No nos interesan las medidas anteriores a las reformas ya que no existen.
above y basement los quitamos ya que living es la suma de estas, y en el precio influyen los metos cuadrados totales""")


dataframe = dataframe.drop(columns=['sqft_above', 'sqft_basement',
                                    'sqft_living', 'sqft_lot'], axis=1)
print(dataframe.head())
print(dataframe.columns)

print("""Las columnas que tengan que ver con numero de habitaciones, las podemos agrupar para simplificar los cálculos
Creamos una nueva columna llamada habitaciones que será la suma de las columnas bedrooms y bathrooms""")


dataframe['rooms'] = 0
for i in range(len(dataframe['bedrooms'])):
    dataframe['rooms'][i] = int(
        dataframe['bedrooms'][i]) + int(dataframe['bathrooms'][i])

print("Exportamos el csv organizado y limpio")


dataframe.to_csv('data/regression_data_clean.csv', index=False)

print("Una vez hecha la limpieza, vamos a pasar a hacer la regresión")


print("Cargamos el csv limpio")


dataframe_regresion = pd.read_csv(
    'data/regression_data_clean.csv', delimiter=',')

print("How many rows of data do you have?")


print(dataframe_regresion.shape)

print("""Find the unique values of the following columns:
    - What are the unique values in the column `bedrooms`?
    - What are the unique values in the column `bathrooms`?
    - What are the unique values in the column `floors`?
    - What are the unique values in the column `condition`?
    - What are the unique values in the column `grade`?
""")


valores_unicos = ['bedrooms', 'bathrooms', 'floors', 'condition', 'grade']

for i in valores_unicos:
    print(dataframe_regresion[i].unique())

print("Arrange the data in decreasing order by the price of the house. Return only the IDs of the top 10 most expensive houses in your data.")


dataframe_regresion = dataframe_regresion.sort_values(
    by='price', ascending=False)
print(dataframe_regresion['id'].head(10))

print("What is the average price of all the properties in your data?")


print(dataframe_regresion['price'].mean())

print("""In this exercise use a simple `groupby` to check the properties of some of the categorical variables in our data

    - What is the average price of the houses grouped by bedrooms? The returned result should have only two columns: `Bedrooms` and `Average price`.
    - What is the average `sqft_living` of the houses grouped by bedrooms? The returned result should have only two columns, `Bedrooms` and `Average_sqft_living`.
    - What is the average price of the houses with a waterfront and without a waterfront? The returned result should have only two columns, `Waterfront` and `Average_price`.
    - Is there any correlation between the columns `condition` and `grade`? Also, create a plot to visually check if there is a positive correlation or negative correlation or no correlation between both variables.
    - Get the number of houses in each category (ie number of houses for a given `condition`) to assess if that category is well represented in the dataset to include it in your analysis. For eg. If the category is under-represented as compared to other categories, ignore that category in this analysis""")


print(dataframe_regresion.groupby('bedrooms')['price'].mean())
print(dataframe_regresion.groupby('bedrooms')['sqft_living15'].mean())
print(dataframe_regresion.groupby('waterfront')['price'].mean())
print(dataframe_regresion.groupby('condition')['grade'].mean())
dataframe_regresion.groupby('condition')['grade'].mean().plot(kind='bar')

categorias = ["id", "bedrooms", "bathrooms", "floors", "waterfront", "view",
              "condition", "grade", "yr_built", "sqft_living15", "sqft_lot15", "price", "rooms"]
for i in categorias:
    print(dataframe_regresion[i].value_counts())

print("""One of the customers is only interested in the following houses:

    - Number of bedrooms either 3 or 4
    - Bathrooms more than 3
    - One Floor
    - No waterfront
    - Condition should be 3 at least
    - Grade should be 5 at least
    - Price smaller than 300000""")


data_cliente = dataframe[(dataframe['bedrooms'] == 3) | (dataframe['bedrooms'] == 4) & (dataframe['bathrooms'] > 3) & (dataframe['floors'] == 1) & (dataframe['waterfront'] == 0) & (dataframe['condition'] >= 3) & (dataframe['grade'] >= 5) & (dataframe['price'] < 300000)]

print(data_cliente)

print("Your manager wants to find out the list of properties whose prices are twice more than the average of all the properties in the database. Write code to show them the list of such properties. ")


def manager():
    dataframe_manager = dataframe_regresion[dataframe_regresion['price']
                                            > dataframe_regresion['price'].mean() * 2]
    return dataframe_manager


print(manager())

print("Most customers are interested in p  roperties with three or four bedrooms. What is the difference in average prices of the properties with three and four bedrooms? In this case you can simply use a `groupby` to check the prices for those particular houses")

print(dataframe_regresion.groupby('bedrooms')['price'].mean())

print("What are the different locations where properties are available in your database? (distinct zip codes).")

print(dataframe["zipcode"].unique())

print("Las localizaciones son por la zona de Seattle, Tacoma y alrededores")

print(" Show all the properties that were renovated.")

print(dataframe[dataframe['yr_renovated'] != 0])

print("Provide the details of the property that is the 11th most expensive property in your database.")

print(dataframe_regresion.sort_values(by='price', ascending=False).iloc[10])

# Se pone 10 porque el 11 es el 10 en python ya que empieza en 0

print("Una vez terminadas las preguntas y el data cleaning, pasamos al modelo de regresión")

plt.figure(figsize=(15, 10))
sns.set(style='white')
mask=np.triu(np.ones_like(dataframe_regresion.corr(), dtype=bool))
cmap=sns.diverging_palette(0, 10, as_cmap=True)
sns.heatmap(dataframe_regresion.corr(),
           mask=mask,
          cmap=cmap,
          center=0,
          square=True,
          annot=True,
          linewidths=0.5,
          cbar_kws={'shrink': 0.5})
plt.savefig("media/1st_correlation_regresion.jpg")
plt.show()