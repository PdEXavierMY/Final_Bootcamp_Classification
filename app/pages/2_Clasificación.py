import streamlit as st
import pandas as pd
import pathlib
import sys

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