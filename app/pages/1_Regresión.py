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

fig = plt.figure(figsize=(15, 10))

sns.set(style='white')

mask=np.triu(np.ones_like(dataframe_regresion.corr(), dtype=bool))

cmap=sns.diverging_palette(0, 10, as_cmap=True)


sns.heatmap(dataframe_regresion.corr(),
           mask=mask,
          cmap=cmap,
          center=0,
          square=True,
          annot=True,
          linewidths=0.5,
          cbar_kws={'shrink': 0.5})
plt.savefig("media/1st_correlation_regresion.jpg")
plt.show()
st.pyplot(fig)