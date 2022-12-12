import streamlit as st
import pandas as pd
import pathlib
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score


# accedemos al directorio padre
path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(path))

# importamos el codigo de analysis/code_classification.py
from analysis.code_classification import *

st.title('Modelo de clasificación')

st.write('''
    Aquí mostraremos los resultados del modelo de clasificación.
''')

lista = data.select_dtypes(include=['int64', 'float64']).columns; print(lista)
corr = data[lista].corr()
fig = plt.figure(figsize=(30, 20))
sns.heatmap(corr, annot=True)
st.pyplot(fig)
#Vemos que no hay ninguna variable que esté muy correlacionada con otra, por lo que no es necesario eliminar ninguna columna basandose en este criterio.
fig2 = plt.figure(figsize=(30, 20))
sns.heatmap(confusion_matrix(y_train, log.predict(X_train)), annot=True)
plt.title('Confusion Matrix Train')
st.pyplot(fig2)
fig3 = plt.figure(figsize=(30, 20))
sns.heatmap(confusion_matrix(y_test, log.predict(X_test)), annot=True)
plt.title('Confusion Matrix Test')
st.pyplot(fig3)