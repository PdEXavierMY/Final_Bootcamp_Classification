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

#ahora convertimos a numericos el resto de valores con get_dummies
data_dummy = pd.get_dummies(data, drop_first=True)
print(data_dummy.info(), "\n")

#todo perfect

#vamos a sacar en orden descendente los 10 clientes con mayor balance medio con su balance medio
data = data_dummy.sort_values(by='average_balance', ascending=False)
print(data[['customer_number', 'average_balance']].head(10), "\n")

#vamos a sacar la media de la columna average_balance, es decir, la media del balance medio de los clientes
print(data['average_balance'].mean(), "\n")

''' - What is the average balance of the customers grouped by `Income Level`? The returned result should have only two columns, `Income` and `Average Balance` of the customers.'''
#vamos a sacar la media del balance medio de los clientes agrupados por nivel de ingresos
print(data.groupby('income_level')['average_balance'].mean(), "\n")

''' - What is the average balance of the customers grouped by `number_of_bank_accounts_open`? The returned result should have only two columns, `number_of_bank_aaccounts_open` and `Average Balance` of the customers. '''
#vamos a sacar la media del balance medio de los clientes agrupados por número de cuentas bancarias abiertas
print(data.groupby('number_of_bank_accounts_open')['average_balance'].mean(), "\n")

'''- What is the average number of credit cards held by customers for each of the credit card ratings? The returned result should have only two columns, `rating` and `average number of credit cards`.'''
#vamos a sacar la media del número de tarjetas de crédito que tienen los clientes agrupados por rating de tarjetas de crédito
print(data.groupby('credit_rating')['credits_cards_held'].mean(), "\n")

'''  - Is there any correlation between the columns `credit_cards_held` and `number_of_bank_accounts_open`? You can analyze this by grouping the data by one of the variables and then aggregating the results of the other column. Visually check if there is a positive correlation or negative correlation or no correlation between the variables.
'''
#vamos a sacar la media del número de tarjetas de crédito que tienen los clientes agrupados por número de cuentas bancarias abiertas
print(data.groupby('number_of_bank_accounts_open')['credits_cards_held'].mean(), "\n")

'''- Check the number of customers in each category (ie number of credit cards held) to assess if that category is well represented in the dataset to include it in your analysis. For eg. If the category is under-represented as compared to other categories, ignore that category in this analysis'''
#vamos a sacar el número de clientes en cada categoría (es decir, número de tarjetas de crédito que tienen) para evaluar si esa categoría está bien representada en el conjunto de datos para incluirla en su análisis. Por ejemplo. Si la categoría está subrepresentada en comparación con otras categorías, ignore esa categoría en este análisis
print(data.groupby('credits_cards_held')['customer_number'].count(), "\n")