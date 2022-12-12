import pandas as pd
import streamlit as st
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
from statsmodels.api import add_constant, OLS
from statsmodels.formula.api import ols
import pylab as plt
import seaborn as sns
import warnings
import sys
import pathlib
# accedemos al directorio padre
path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(path))

# importamos el codigo de analysis/code_regression.py
from analysis.code_regression import *

st.title('Modelo de regresión:')

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

def plot_regression_model(x,y):
    global figr2
    x_const = add_constant(x) # add a constant to the model
    modelo = OLS(y, x_const).fit() # fit the model
    pred = modelo.predict(x_const) # make predictions
    print(modelo.summary()) # print the summary
    try:
        const = modelo.params[0] # create a variable with the value of the constant given by the summary
        coef = modelo.params[1] # create a variable with the value of the coef given by the summary

        x_l=np.linspace(x.min(), x.max(), 50)
        y_l= coef*x_l + const # function of the line

        figr2 = plt.figure(figsize=(10, 10))

        # plot the line
        plt.plot(x_l, y_l, label=f'{x.name} vs {y.name}={coef}*{x.name}+{const}');

        # data
        plt.scatter(x, y, marker='x', c='g', label=f'{x.name} vs {y.name}');

        plt.title('Regresion lineal')
        plt.xlabel(f'{x.name}')
        plt.ylabel(f'{y.name}')
        plt.legend()
        st.pyplot(figr2)
        plt.show()
        return modelo
    except:
        print('No se puede imprimir la recta de regresión para modelos multivariable')
        st.pyplot(figr2)
        plt.show()
        return modelo

print("Vamos a hacer un modelo de regresión lineal para predecir el precio de las casas\n")
x = dataframe_regresion['sqft_living15']
y = dataframe_regresion['price']
modelo = plot_regression_model(x,y)
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

fig3r = plt.figure(figsize=(20,20))
sns.histplot([np.exp(y_pred), y_test])
plt.title("Histograma de los datos reales y los datos predichos")
st.pyplot(fig3r)