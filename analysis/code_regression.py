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

print(Fore.CYAN + "Cargamos el csv con la librería pandas")
print(Fore.RESET)
dataframe = pd.read_csv('data/regression_data.csv', delimiter=';')
print(dataframe.head())

print(Fore.CYAN + "Examinamos el csv")
print(Fore.RESET)
print('Las columnas son:\n')
print(dataframe.columns)
print('La descripción del csv es:\n')
print(dataframe.describe())

print(Fore.CYAN + """Observaciones:
Las columnas no numéricas son : date, bathrooms, floor, lat, long
Como queremos predecir el precio, la columna que va a ir por separado es price
Para predecir el precio necesitamos: bedrooms, bathrooms, sqft, floors, condiciones, grade, year_built, year_renovate, waterview
lat, long y zipcode no nos influye ya que estan todas las casas en la misma zona de Seattle, Tacoma y alrededores, gracias a Google Maps""")
print(Fore.RESET)


print(Fore.CYAN + "Pasamos a la limpieza de los datos")
print(Fore.RESET)

print(Fore.CYAN + "Hacemos una copia de los datos")
print(Fore.RESET)

copia_dataframe = dataframe.copy()

print(Fore.CYAN + "Los datos que son floats que no tengan sentido, como 2,5 baños, vamos a pasarlos a integers:")
print(Fore.RESET)


def redondear_datos(columna):
    for i in range(len(dataframe)):
        if ',' in dataframe[columna][i]:
            dataframe[columna][i] = float(
                dataframe[columna][i].replace(',', '.'))
            dataframe[columna][i] = round(dataframe[columna][i])
    return dataframe


datos_a_redondear = ['bathrooms', 'floors']

for i in datos_a_redondear:
    redondear_datos(i)
    print(dataframe[i])

print(Fore.CYAN + "Ahora vamos a quitar las columnas que no nos hacen falta: lat, long, date, zipcode")
print(Fore.RESET)

dataframe = dataframe.drop(columns=['date', 'lat', 'long'], axis=1)

print(Fore.CYAN + """Para facilitar el análisis, vamos a agrupar todas las columnas que contengan mediciones en metros cuadrados, creando así una única columna con todos los metros cuadrados de la casa.
Tendremos al final dos columnas: la medida del terreno (lot) y la medida de la casa (living)
No nos interesan las medidas anteriores a las reformas ya que no existen.
above y basement los quitamos ya que living es la suma de estas, y en el precio influyen los metos cuadrados totales""")
print(Fore.RESET)

dataframe = dataframe.drop(columns=['sqft_above', 'sqft_basement',
                                    'sqft_living', 'sqft_lot'], axis=1)
print(dataframe.head())
print(dataframe.columns)

print(Fore.CYAN + """Las columnas que tengan que ver con numero de habitaciones, las podemos agrupar para simplificar los cálculos
Creamos una nueva columna llamada habitaciones que será la suma de las columnas bedrooms y bathrooms""")
print(Fore.RESET)

dataframe['rooms'] = 0
for i in range(len(dataframe['bedrooms'])):
    dataframe['rooms'][i] = int(
        dataframe['bedrooms'][i]) + int(dataframe['bathrooms'][i])

print(Fore.CYAN + "Exportamos el csv organizado y limpio")
print(Fore.RESET)

dataframe.to_csv('data/regression_data_clean.csv', index=False)

print(Fore.CYAN + "Una vez hecha la limpieza, vamos a pasar a hacer la regresión")
print(Fore.RESET)

print(Fore.CYAN + "Cargamos el csv limpio")
print(Fore.RESET)

dataframe_regresion = pd.read_csv(
    'data/regression_data_clean.csv', delimiter=',')

print(Fore.GREEN + "How many rows of data do you have?")
print(Fore.RESET)

print(dataframe_regresion.shape)

print(Fore.GREEN + """Find the unique values of the following columns:
    - What are the unique values in the column `bedrooms`?
    - What are the unique values in the column `bathrooms`?
    - What are the unique values in the column `floors`?
    - What are the unique values in the column `condition`?
    - What are the unique values in the column `grade`?
""")
print(Fore.RESET)

valores_unicos = ['bedrooms', 'bathrooms', 'floors', 'condition', 'grade']

for i in valores_unicos:
    print(dataframe_regresion[i].unique())

print(Fore.GREEN + "Arrange the data in decreasing order by the price of the house. Return only the IDs of the top 10 most expensive houses in your data.")
print(Fore.RESET)

dataframe_regresion = dataframe_regresion.sort_values(
    by='price', ascending=False)
print(dataframe_regresion['id'].head(10))

print(Fore.GREEN + "What is the average price of all the properties in your data?")
print(Fore.RESET)

print(dataframe_regresion['price'].mean())

print(Fore.GREEN + """In this exercise use a simple `groupby` to check the properties of some of the categorical variables in our data

    - What is the average price of the houses grouped by bedrooms? The returned result should have only two columns: `Bedrooms` and `Average price`.
    - What is the average `sqft_living` of the houses grouped by bedrooms? The returned result should have only two columns, `Bedrooms` and `Average_sqft_living`.
    - What is the average price of the houses with a waterfront and without a waterfront? The returned result should have only two columns, `Waterfront` and `Average_price`.
    - Is there any correlation between the columns `condition` and `grade`? Also, create a plot to visually check if there is a positive correlation or negative correlation or no correlation between both variables.
    - Get the number of houses in each category (ie number of houses for a given `condition`) to assess if that category is well represented in the dataset to include it in your analysis. For eg. If the category is under-represented as compared to other categories, ignore that category in this analysis
""")
print(Fore.RESET)

print(dataframe_regresion.groupby('bedrooms')['price'].mean())
print(dataframe_regresion.groupby('bedrooms')['sqft_living15'].mean())
print(dataframe_regresion.groupby('waterfront')['price'].mean())
print(dataframe_regresion.groupby('condition')['grade'].mean())
dataframe_regresion.groupby('condition')['grade'].mean().plot(kind='bar')
plt.show()

categorias = ["id", "bedrooms", "bathrooms", "floors", "waterfront", "view",
              "condition", "grade", "yr_built", "sqft_living15", "sqft_lot15", "price", "rooms"]
for i in categorias:
    print(dataframe_regresion[i].value_counts())

print(Fore.GREEN + """One of the customers is only interested in the following houses:

    - Number of bedrooms either 3 or 4
    - Bathrooms more than 3
    - One Floor
    - No waterfront
    - Condition should be 3 at least
    - Grade should be 5 at least
    - Price smaller than 300000""")
print(Fore.RESET)

def cliente():
    if (dataframe_regresion['bedrooms'] == 3 or dataframe_regresion['bedrooms'] == 4) and dataframe_regresion['bathrooms'] > 3 and dataframe_regresion['floors'] == 1 and dataframe_regresion['waterfront'] == 0 and dataframe_regresion['condition'] >= 3 and dataframe_regresion['grade'] >= 5 and dataframe_regresion['price'] < 300000:
        return dataframe_regresion

print(cliente())
print(Fore.CYAN + "No hay ninguna casa que cumpla con los requisitos del cliente")

print(Fore.GREEN + "Your manager wants to find out the list of properties whose prices are twice more than the average of all the properties in the database. Write code to show them the list of such properties. ")
print(Fore.RESET)


def manager():
    dataframe_manager = dataframe_regresion[dataframe_regresion['price']
                                            > dataframe_regresion['price'].mean() * 2]
    return dataframe_manager


print(manager())

print(Fore.GREEN + "Most customers are interested in p  roperties with three or four bedrooms. What is the difference in average prices of the properties with three and four bedrooms? In this case you can simply use a `groupby` to check the prices for those particular houses") ; print(Fore.RESET)

print(dataframe_regresion.groupby('bedrooms')['price'].mean())

print(Fore.GREEN + "What are the different locations where properties are available in your database? (distinct zip codes).") ; print(Fore.RESET)

print(Fore.CYAN + "Las localizaciones son por la zona de Seattle, Tacoma y alrededores") ; print(Fore.RESET)

print(Fore.GREEN + " Show all the properties that were renovated.") ; print(Fore.RESET)

print(dataframe[dataframe['yr_renovated'] != 0])

print(Fore.GREEN + "Provide the details of the property that is the 11th most expensive property in your database.") ; print(Fore.RESET)

print(dataframe_regresion.sort_values(by='price', ascending=False).iloc[11])


