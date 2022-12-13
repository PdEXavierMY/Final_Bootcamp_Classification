from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
from statsmodels.api import add_constant, OLS
from statsmodels.formula.api import ols
import pylab as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Cargamos el dataset


dataframe_regresion = pd.read_csv("data/regression_data_clean.csv")
plt.figure(figsize=(15, 10))
sns.set(style='white')
mask = np.triu(np.ones_like(dataframe_regresion.corr(), dtype=bool))
cmap = sns.diverging_palette(0, 10, as_cmap=True)
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


def plot_regression_model(x, y):
    x_const = add_constant(x)  # add a constant to the model
    modelo = OLS(y, x_const).fit()  # fit the model
    pred = modelo.predict(x_const)  # make predictions
    print(modelo.summary())  # print the summary
    try:
        # create a variable with the value of the constant given by the summary
        const = modelo.params[0]
        # create a variable with the value of the coef given by the summary
        coef = modelo.params[1]

        x_l = np.linspace(x.min(), x.max(), 50)
        y_l = coef*x_l + const  # function of the line

        plt.figure(figsize=(10, 10))

        # plot the line
        plt.plot(
            x_l, y_l, label=f'{x.name} vs {y.name}={coef}*{x.name}+{const}')

        # data
        plt.scatter(x, y, marker='x', c='g', label=f'{x.name} vs {y.name}')

        plt.title('Regresion lineal')
        plt.xlabel(f'{x.name}')
        plt.ylabel(f'{y.name}')
        plt.legend()
        plt.show()
        return modelo
    except:
        print('No se puede imprimir la recta de regresión para modelos multivariable')
        plt.show()
        return modelo

# Ahora vamos a hacer un modelo de regresión lineal para predecir el precio de las casas

print("Vamos a hacer un modelo de regresión lineal para predecir el precio de las casas\n")
x = dataframe_regresion['sqft_living15']
y = dataframe_regresion['price']
modelo = plot_regression_model(x, y)

# Dividimos el dataset en train y test

x = dataframe_regresion[['sqft_living15']]
y = dataframe_regresion['price']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)


# Ahora normalizamos los datos

pt = PowerTransformer()
pt.fit(x_train)
with open("scalers/scalers.plk", "wb") as f:
    pickle.dump(pt, f)
# Los guardamos en un archivo pickle para poder usarlos en el futuro

# Normalizamos los datos para que se asmejen a una distribución normal

x_train_scaled = pt.transform(x_train)
x_test_scaled = pt.transform(x_test)

y_train_scaled = np.log(y_train)

x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns)

x_train_scaled


# Ahora vamos a crear el modelo de regresión lineal

lr = LinearRegression()
lr.fit(x_train_scaled, y_train_scaled)

# Ahora vamos a predecir los datos
y_pred = lr.predict(x_test_scaled)

# Calculamos el R2

r2_score(y_test, np.exp(y_pred))
print("El R2 es de: ", r2_score(y_test, np.exp(y_pred)))

# Calculando el R2, hemos obtenido un valor muy bajo debido a que el modelo no es muy bueno, y tiene mucho error

# Ahora vamos a guardar el modelo

with open("models/modelo_regresion.plk", "wb") as f:
    pickle.dump(lr, f)

# Ahora vamos a hacer un histograma de los datos reales y los datos predichos

plt.figure(figsize=(15, 15))
sns.histplot([np.exp(y_pred), y_test])
plt.title("Histograma de los datos reales y los datos predichos")
plt.savefig("media/histograma_datos_reales_predichos.png") # Guardamos la imagen
plt.show()