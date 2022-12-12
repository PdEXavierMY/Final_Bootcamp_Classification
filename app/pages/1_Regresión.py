import streamlit as st
import pandas as pd
import pathlib
import sys

# accedemos al directorio padre
path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(path))

# importamos el codigo de analysis/code_regression.py
from analysis.code_regression import *

st.title('Modelo de regresión')

st.write('''
    Aquí mostraremos los resultados del modelo de regresión.
''')

st.code(pathlib.Path('analysis/code_regression.py').read_text())

fig = dataframe_regresion.groupby('condition')['grade'].mean().plot(kind='bar')
st.pyplot(fig)