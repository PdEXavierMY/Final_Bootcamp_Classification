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

#como no tenemos un numero demasiado grande de datos vamos a hacer un oversampling en los datos train
#vamos a probar con ambos métodos de oversampling: SMOTE y random oversampling
smote = SMOTE()
rov = RandomOverSampler()
fig6 = plt.figure(figsize=(15, 10))
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(X_train_sm.shape, y_train_sm.shape, '\n')
print(y_train_sm.value_counts(), '\n')
sns.histplot(y_train_sm)
st.pyplot(fig6)
plt.show()

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
fig7 = plt.figure(figsize=(15, 10))
sns.heatmap(confusion_matrix(y_train_sm, log.predict(X_train_sm)), annot=True);
plt.title('Confusion Matrix Train')
st.pyplot(fig7)
plt.show()
fig8 = plt.figure(figsize=(15, 10))
sns.heatmap(confusion_matrix(y_test, log.predict(X_test)), annot=True)
plt.title('Confusion Matrix Test')
st.pyplot(fig8)
plt.show()

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

    figcorr= plt.figure(figsize=(20,15))
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
    
    if save:
        try:
            plt.savefig(f'./media/{title}.jpg')
        except:
            destino = input('No exite la carpeta de destino, introduce un nombre para la carpeta de destino: ')
            os.mkdir(destino)
            plt.savefig(f'{destino}/{title}.jpg')
    st.pyplot(figcorr)
    plt.show()

print_heatmap_corr(data_dummy_pos_coef)

'''Vemos que la las variables independientes no tienen mucha correlación con nuestra variable dependiente, esto quiere decir que la solución al problema es compleja y hay que tratarla con cuidado, y también puede darnos una indicación de que los resultados que podemos esperar de los modelos no van ha ser muy buenos, pero tenemos que tratar de hacer todo lo posible para que estos sean lo más altos posibles. Respecto a la correlación entre las variables independientes vemos que salvo TotalCharges no hay excesiva colinealidad entre nuestras variables, por lo que nos quedaremos con ellas.

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