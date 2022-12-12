#Vamos a realizar un modelo de classification en los datos de unos clientes de banco 
#para estudiar y predecir el comportamiento de los clientes en función de sus características

#primero importamos las librerías necesarias
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import streamlit as st

import pylab as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score

import warnings
warnings.filterwarnings('ignore')

# Importamos los datos
data = pd.read_csv('./data/creditcardmarketing.csv')
print(data) #vemos que se ha importado correctamente
print(data.head(10), "\n") #vemos las 10 primeras filas del dataset

#sacamos la información de los datos para revisar si necesitamos hacer algún ajuste a nuestros dataset
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

#vamos cambiar a numericos los valores de las columnas offer_acepted,reward

data['offer_acepted'] = data['offer_acepted'].map({'No': 0, 'Yes': 1})
data['reward'] = data['reward'].map({'Air Miles': 0, 'Cash Back': 1, 'Points': 2})

#comprobamos que ha funcionado printeando los valores unicos de nuevo
print(data['offer_acepted'].unique(), "\n")
print(data['reward'].unique(), "\n")

#todo perfect(las demás columnas no numéricas las vamos a ver más adelante)

#vamos a sacar en orden descendente los 10 clientes con mayor balance medio con su balance medio
data1 = data.sort_values(by='average_balance', ascending=False)
print(data1[['customer_number', 'average_balance']].head(10), "\n")

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
data_interesante = data[(data['credit_rating'] == 'Medium') | (data['credit_rating'] == 'High') & (data['credits_cards_held'] <= 2) & (data['own_home'] == 'Yes') & (data['household_size'] >= 3)]
print(data_interesante, "\n")

#sacamos los clientes con un average_balance < a la media de la columna average_balance
data_average_balance = data[data['average_balance'] < data['average_balance'].mean()]
data_average_balance.sort_values(by='average_balance', ascending=True)
print(data_average_balance[['customer_number', 'average_balance']], "\n")

#sacamos el numero de clientes que aceptan la oferta y el numero de clientes que no aceptan la oferta
print(data.groupby('offer_acepted')['customer_number'].count(), "\n")
#recordar 0 = oferta no aceptada y 1 = oferta aceptada

#sacamos los datos con credit rating = high or low
data_credit_rating = data[(data['credit_rating'] == 'High') | (data['credit_rating'] == 'Low')]
#sacamos la media de el average balance de los clientes con credit rating = high or low
print(data_credit_rating.groupby('credit_rating')[['credit_rating', 'average_balance']].mean(), "\n")
print("La diferencia entre la media de balances de aquellos clientes con un rating alto y uno bajo es ", (data_credit_rating.groupby('credit_rating')['average_balance'].mean().iloc[0]-data_credit_rating.groupby('credit_rating')['average_balance'].mean().iloc[1]), "\n")

#vamos a ver los tipos de mailer_type que hay y con cuantos clientes se han usado
print(data.groupby('mailer_type')['customer_number'].count(), "\n")

#por ultimo vamos a sacar al onceavo cliente con menos balanceq1 en nuestro dataset
data2 = data.sort_values(by='balanceq1', ascending=True)
print(data2.iloc[10], "\n")

#vamos ahora a terminar con las columnas no numéricas
print(data.dtypes, "\n")
#las columnas mailer_type, overdraft_protection, own_home, balanceq1, balanceq2, balanceq3 y balanceq4 no nos sirven para el modelo de clasificacion asi que las vamos a eliminar
data = data.drop(['mailer_type', 'overdraft_protection', 'own_home', 'balanceq1', 'balanceq2', 'balanceq3', 'balanceq4'], axis=1)
#ahora con las dos columnas que nos quedan vamos a cambiarlas a numericas manualmente
data['credit_rating'] = data['credit_rating'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['income_level'] = data['income_level'].map({'Low': 0, 'Medium': 1, 'High': 2})
#volvemos a ver los tipos de datos
print(data.dtypes, "\n")
#ya tenemos todas las columnas numericas
#finalmente vamos a guardar el dataset en un nuevo csv
data.to_csv('data/creditcardmarketing_clean.csv', index=False)

#con el dataset ya limpio, volvemos a ver la matriz de correlacion
plt.figure(figsize=(15, 10))
sns.set(style='white')
mask=np.triu(np.ones_like(data.corr(), dtype=bool))
cmap=sns.diverging_palette(0, 10, as_cmap=True)
sns.heatmap(data.corr(),
          mask=mask,
          cmap=cmap,
          center=0,
          square=True,
          annot=True,
          linewidths=0.5,
          cbar_kws={'shrink': 0.5})
#plt.show()




#vamos a empezar con el modelo de clasificacion
data_dummy = pd.get_dummies(data, drop_first = True)
#primero separaremos los datos en train y test
#Seleccionamos 80% de los datos para training y 20% para testing

X = data_dummy.drop('offer_acepted', axis = True)
y = data_dummy['offer_acepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
'''minmax = MinMaxScaler().fit(X_train)
X_train = minmax.transform(X_train)
X_test = minmax.transform(X_test)'''
print(y_test.value_counts(normalize = True), y_train.value_counts(normalize = True))

log = LogisticRegression()
log.fit(X_train, y_train)
score_train = log.score(X_train, y_train)
score_test = log.score(X_test, y_test)
        
res_num = {'l_train_score': score_train, 'l_test_score': score_test}
print(res_num)
#Parece ser que tenemos un buen resultado pero el accuracy en modelos de clasificación no es la mejor métrica de evaluación

log.fit(X_train, y_train)
score_train = log.score(X_train, y_train)
score_test = log.score(X_test, y_test)
precision_train = precision_score(y_train, log.predict(X_train))
precision_test = precision_score(y_test, log.predict(X_test))
recall_train = recall_score(y_train, log.predict(X_train))
recall_test = recall_score(y_test, log.predict(X_test))
f1_train = f1_score(y_train, log.predict(X_train))
f1_test = f1_score(y_test, log.predict(X_test))
        
res_num = {'l_train_score': score_train,
           'l_test_score': score_test,
           'l_train_precision': precision_train,
           'l_test_precision': precision_test,
           'l_train_recall': recall_train,
           'l_test_recall': recall_test,
           'l_f1_train': f1_train,
           'l_f1_test': f1_test}
        
sns.heatmap(confusion_matrix(y_train, log.predict(X_train)), annot=True)
plt.title('Confusion Matrix Train')
#plt.show()
sns.heatmap(confusion_matrix(y_test, log.predict(X_test)), annot=True)
plt.title('Confusion Matrix Test')
#plt.show()
print(res_num)


'''lg = LogisticRegression(max_iter = 2000)
lg.fit(X_train, y_train)
pred = lg.predict(X_test)


     

acc = accuracy_score(y_test, pred)
print(f'Accuracy lg: {acc}')

cm = confusion_matrix(y_test, pred)

plt.figure(figsize = (15, 20))
sns.heatmap(cm, annot = True)
plt.show()'''