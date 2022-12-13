import streamlit as st
import pathlib
import sys
import pandas as pd
import os
pd.set_option('display.max_columns', None)
import numpy as np
import pylab as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import warnings
warnings.filterwarnings('ignore')


# accedemos al directorio padre
path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(path))

# importamos el codigo de analysis/code_classification.py
from analysis.code_classification import *

st.title('Modelo de clasificación:')

st.write('''
    Aquí mostraremos los resultados del modelo de clasificación.
''')
lista = data.select_dtypes(include=['int64', 'float64']).columns; print(lista)
corr = data[lista].corr()
fig1 = plt.figure(figsize=(30, 20))
sns.heatmap(corr, annot=True)
st.pyplot(fig1)
st.write('''Con los datos limpios, mostramos la matriz de correlación''')
fig2 = plt.figure(figsize=(15, 10))
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
st.pyplot(fig2)
#Vemos que no hay ninguna variable que esté muy correlacionada con otra, por lo que no es necesario eliminar ninguna columna basandose en este criterio.

data_dummy = pd.get_dummies(data, drop_first = True)
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
fig3 = plt.figure(figsize=(15, 10))  
sns.heatmap(confusion_matrix(y_train, log.predict(X_train)), annot=True)
plt.title('Confusion Matrix Train')
st.pyplot(fig3)
plt.show()
fig4 = plt.figure(figsize=(15, 10))
sns.heatmap(confusion_matrix(y_test, log.predict(X_test)), annot=True)
plt.title('Confusion Matrix Test')
st.pyplot(fig4)
plt.show()
print(res_num, '\n')
#nuestra precision es bastante mala, vamos a probar con un modelo de arbol de decision
fig5 = plt.figure(figsize=(15, 10))
#vamos a ver el balanceo de los datos
sns.countplot(data_dummy.offer_acepted)
# mostramos el gráfico con streamlit
st.pyplot(fig5)
plt.show()