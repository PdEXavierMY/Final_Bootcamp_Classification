#Vamos a realizar un modelo de classification en los datos de unos clientes de banco 
# para estudiar y predecir el comportamiento de los clientes en función de sus características

#primero importamos las librerías necesarias
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import streamlit as st

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

#Vemos que no hay ninguna variable que esté muy correlacionada con otra, por lo que no es necesario eliminar ninguna columna basandose en este criterio.

#vamos con los valores nulos. Como son numericos los rellenaremos con la media de la columna
#las columnas que vamos a rellenar con la media son las siguientes: average_balance,balanceq1,balanceq2,balanceq3,balanceq4

data['average_balance'].fillna(data['average_balance'].mean(), inplace=True)
data['balanceq1'].fillna(data['balanceq1'].mean(), inplace=True)
data['balanceq2'].fillna(data['balanceq2'].mean(), inplace=True)
data['balanceq3'].fillna(data['balanceq3'].mean(), inplace=True)
data['balanceq4'].fillna(data['balanceq4'].mean(), inplace=True)


#comprobamos que no hay más valores nulos
print(data.isnull().sum(), "\n")

#ahora vamos a gestionar las columnas no numéricas
#vamos a ver los valores unicos de estas columnas: offer_acepted,reward,mailer_type,credits_cards_held,household_size

print(data['offer_acepted'].unique(), "\n")
print(data['reward'].unique(), "\n")
print(data['mailer_type'].unique(), "\n")
print(data['credits_cards_held'].unique(), "\n")
print(data['household_size'].unique(), "\n")

#vamos cambiar a numericos los valores de las columnas offer_acepted,reward,mailer_type

data['offer_acepted'] = data['offer_acepted'].map({'No': 0, 'Yes': 1})
data['reward'] = data['reward'].map({'Air Miles': 0, 'Cash Back': 1, 'Points': 2})
data['mailer_type'] = data['mailer_type'].map({'Letter': 0, 'Postcard': 1})

#comprobamos que ha funcionado printeando los valores unicos de nuevo
print(data['offer_acepted'].unique(), "\n")
print(data['reward'].unique(), "\n")
print(data['mailer_type'].unique(), "\n")

#todo perfect(las demás columnas no numéricas no las vamos a cambiar porque no es necesario)

#vamos a sacar en orden descendente los 10 clientes con mayor balance medio con su balance medio
data = data.sort_values(by='average_balance', ascending=False)
print(data[['customer_number', 'average_balance']].head(10), "\n")

#vamos a sacar la media de la columna average_balance, es decir, la media del balance medio de los clientes
print(data['average_balance'].mean(), "\n")

#vamos a sacar la media del balance medio de los clientes agrupados por nivel de ingresos
print(data.groupby('income_level')[['income_level', 'average_balance']].mean(), "\n")

#vamos a sacar la media del balance medio de los clientes agrupados por número de cuentas bancarias abiertas
print(data.groupby('banks_accounts_open')['average_balance'].mean(), "\n")

#vamos a sacar la media del número de tarjetas de crédito que tienen los clientes agrupados por rating de tarjetas de crédito
print(data.groupby('credit_rating')[['credit_rating', 'credits_cards_held']].mean(), "\n")

#vamos a sacar la media del número de tarjetas de crédito que tienen los clientes agrupados por número de cuentas bancarias abiertas
print(data.groupby('banks_accounts_open')[['banks_accounts_open', 'credits_cards_held']].mean(), "\n")
#estos datos nos aproximan la correlacion entre el numero de cuentas bancarias abiertas y el numero de tarjetas de credito que tienen los clientes,
#pero para conocer con más precisión el resultado podemos fijarnos en la matriz de correlacion del principio
#conclusión: no hay correlacion significativa entre el numero de cuentas bancarias abiertas y el numero de tarjetas de credito que tienen los clientes

#vamos a sacar el número de clientes en cada categoría (es decir, número de tarjetas de crédito que tienen) para evaluar si esa categoría está bien representada en el conjunto de datos para incluirla en su análisis. Por ejemplo. Si la categoría está subrepresentada en comparación con otras categorías, ignoraremos esa categoría en este análisis
print(data.groupby('credits_cards_held')['customer_number'].count(), "\n")

#ahora vamos a filtrar y sacar los clientes que tienen credit_rating = medium o high, credit_cards_held = 2 o menos, own_home = Yes y household_size = 3 o más
data_interesante = data[(data['credit_rating'] == 'medium') | (data['credit_rating'] == 'high') & (data['credits_cards_held'] <= 2) & (data['own_home'] == 'Yes') & (data['household_size'] >= 3)]
print(data_interesante, "\n")