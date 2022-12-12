import streamlit as st
import pandas as pd
import pathlib
import sys
import plotly.figure_factory as ff
import plotly.graph_objects as go

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
fig = plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True)
st.pyplot(fig)