#Vamos a realizar un modelo de classification en los datos de unos clientes de banco 
#para estudiar y predecir el comportamiento de los clientes en función de sus características

#primero importamos las librerías necesarias
import pandas as pd
import os
pd.set_option('display.max_columns', None)
import numpy as np
import streamlit as st

import pylab as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score

from imblearn.over_sampling import SMOTE

import pickle

import warnings
warnings.filterwarnings('ignore')


#vamos a empezar con el modelo de clasificacion
data_dummy = pd.read_csv('./data/creditcardmarketing_clean.csv')
#primero separaremos los datos en train y test
#Seleccionamos 80% de los datos para training y 20% para testing

X = data_dummy.drop('offer_acepted', axis = True)
y = data_dummy['offer_acepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
minmax = MinMaxScaler().fit(X_train)
X_train = minmax.transform(X_train)
X_test = minmax.transform(X_test)
print(y_test.value_counts(normalize = True), '\n', y_train.value_counts(normalize = True), '\n')

log = LogisticRegression(max_iter=2000)
log.fit(X_train, y_train)
score_train = log.score(X_train, y_train)
score_test = log.score(X_test, y_test)
        
res_num = {'l_train_score': score_train, 'l_test_score': score_test}
print(res_num, '\n')
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
plt.show()
sns.heatmap(confusion_matrix(y_test, log.predict(X_test)), annot=True)
plt.title('Confusion Matrix Test')
plt.show()
print(res_num, '\n')
'''Nuestra precision es horrible, vamos a ver que pasa con los datos. Para ello,
vamos a estudiar el balanceo de los datos para hacernos una idea de que es lo que falla'''
sns.countplot(data_dummy.offer_acepted)
plt.show()

'''Los numeros estan muy desbalanceados, así que vamos a evaluar nuestras opciones.
Como no tenemos un número extremadamente grande de datos vamos a hacer un oversampling en los datos train,
en este caso decantándonos por el método de oversampling: SMOTE'''
smote = SMOTE()

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(X_train_sm.shape, y_train_sm.shape, '\n')
print(y_train_sm.value_counts(), '\n')
sns.histplot(y_train_sm)
plt.show()
'''Balanceamos(forzamos) los datos'''

log.fit(X_train_sm, y_train_sm)
score_train_sm = log.score(X_train_sm, y_train_sm)
score_test_sm = log.score(X_test, y_test)
precision_train_sm = precision_score(y_train_sm, log.predict(X_train_sm))
precision_test_sm = precision_score(y_test, log.predict(X_test))
recall_train_sm = recall_score(y_train_sm, log.predict(X_train_sm))
recall_test_sm = recall_score(y_test, log.predict(X_test))
f1_train_sm = f1_score(y_train_sm, log.predict(X_train_sm))
f1_test_sm = f1_score(y_test, log.predict(X_test))
        
res_sm = {'le_train_sm_score': score_train_sm,
          'le_test_sm_score': score_test_sm,
          'le_train_precision': precision_train_sm,
          'le_test_precision': precision_test_sm,
          'le_train_recall': recall_train_sm,
          'le_test_recall': recall_test_sm,
          'le_f1_train_sm': f1_train_sm,
          'le_f1_test_sm': f1_test_sm}
        
sns.heatmap(confusion_matrix(y_train_sm, log.predict(X_train_sm)), annot=True);
plt.title('Confusion Matrix Train')
plt.show()
sns.heatmap(confusion_matrix(y_test, log.predict(X_test)), annot=True)
plt.title('Confusion Matrix Test')
plt.show()

print(res_sm, '\n')
'''El modelo ha mejorado un poco y se ha corregido el overfitting. El test ha subido por lo que ya tenemos un dato bajo pero suficiente para poder realizar nuestra predicción. El único problema es la precisión de nuestro modelo, vamos a ver si podemos mejorarla.
Llegados a este punto podemos valorar varias opciones o bien realizar un análisis más profundo de los datos y ver como están correlacionados estos en busca de colinealidad o realizar un estudio de importancia de características. Vamos a probar a ajustar el punto de intersección de la regresión logística y evaluar los coeficientes de cada una de las características y su correlación en busca de colinealidad.'''

print(log.intercept_, '\n')
coefs = dict(zip(list(data_dummy.drop(['offer_acepted'], axis=1).columns),list(log.coef_[0])))
print(coefs, '\n')

'''Vamos a interpretar estos coeficientes, también denominados R statistic.

Un valor positivo significa que al crecer la variable predictora, lo hace la probabilidad de que el evento ocurra. Un valor negativo implica que si la variable predictora decrece, la probabilidad de que el resultado ocurra disminuye. Si una variable tiene un valor pequeño de R entonces esta contribuye al modelo sólo una pequeña cantidad.

De esto podemos extraer que si quitamos todas las columnas con un coeficiente negativo, nuestro modelo podría mejorar.'''

neg_coef = []

for k,v in coefs.items():
    if v < 0:
        neg_coef.append(k)
print(neg_coef, '\n')
data_dummy_pos_coef = data_dummy.drop(neg_coef, axis=1)

def print_heatmap_corr(data:pd.DataFrame, annot:bool=True, cmap:str=None, 
                       mask:bool=True, save:bool=False, title:str=None)->None:

    sns.set(style='white')
    if mask:
        mascara=np.triu(np.ones_like(data.corr(), dtype=bool))
    else:
        mascara = None

    if cmap:
        c_map = sns.color_palette(cmap, as_cmap=True)
    else:
        c_map=sns.diverging_palette(0, 10, as_cmap=True)

    plt.figure(figsize=(20,15))
    p = sns.heatmap(data.corr(),
            mask=mascara,
            cmap=c_map,
            vmax=1,
            center=0,
            square=True,
            linewidth=0.5,
            cbar_kws={'shrink': 0.5},
            annot=annot
           )
    p.set_title(title, fontsize=20)
    plt.show()

print_heatmap_corr(data_dummy_pos_coef)

''''Vemos que las variables independientes no tienen mucha correlación con nuestra variable dependiente, lo que quiere decir que la solución al problema es compleja, y también puede darnos una indicación de que los resultados que podemos esperar de los modelos no van ha ser los mejores. Respecto a la correlación entre las variables independientes vemos no hay excesiva colinealidad entre nuestras variables, por lo que nos quedaremos con ellas.

Como nuestro set de datos es diferente al original debemos de volver a realizar el train_test_split de nuevo, junto con el scaler y el smote para corregir el balanceo.

Declararemos de nuevo nuestras X e y y aplicamos todo el proceso de transformaciones.'''

X = data_dummy_pos_coef.drop(['offer_acepted'], axis=1)
y = data_dummy_pos_coef['offer_acepted']
print(X.shape, y.shape, '\n')
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, '\n')
minmax = MinMaxScaler().fit(X_train)
X_train = minmax.transform(X_train)
X_test = minmax.transform(X_test)

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
print(X_train.shape, y_train.shape, '\n')
print(y_train.value_counts(), '\n')
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
score_train = lr.score(X_train, y_train)
score_test = lr.score(X_test, y_test)
precision_train = precision_score(y_train, lr.predict(X_train))
precision_test = precision_score(y_test, lr.predict(X_test))
recall_train = recall_score(y_train, lr.predict(X_train))
recall_test = recall_score(y_test, lr.predict(X_test))
f1_train = f1_score(y_train, lr.predict(X_train))
f1_test = f1_score(y_test, lr.predict(X_test))
        
res = {'lr_train_score': score_train,
       'lr_test_score': score_test,
       'lr_train_precision': precision_train,
       'lr_test_precision': precision_test,
       'lr_train_recall': recall_train,
       'lr_test_recall': recall_test,
       'lr_f1_train': f1_train,
       'lr_f1_test': f1_test}
        
sns.heatmap(confusion_matrix(y_train, lr.predict(X_train)), annot=True)
plt.title('Confusion Matrix Train')
plt.show()
sns.heatmap(confusion_matrix(y_test, lr.predict(X_test)), annot=True)
plt.title('Confusion Matrix Test')
plt.show()
print(res, '\n')

'''En este nuevo modelo la precisión junto al resto de valores ha disminuido, así que nos quedaremos con el modelo anterior. El modelo escogido nos da cierta información acerca de que clientes nos pueden interesar, pero no es un modelo que nos de una gran precisión, por lo que no podemos confiar en él con plenitud para tomar decisiones.'''


pickle.dump(lr, open('./models/modelo_clasificacion.pkl', 'wb'))