
import streamlit as st
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

    Vamos a hacer un modelo de regresión lineal para predecir el precio de las casas.

''')
st.write('''En primer lugar, cargamos los datos y representamos la matriz de correlación.''')
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

st.pyplot(fig)
st.write('''Como se puede observar en la matriz de correlación no hya ninguna que tenga más de un 0.9 de correlación, por lo tanto no podemos eliminar esas variables, la que más se acerca es la bedrooms con rooms, pero como rooms es la agrupación de bedrooms y bathrooms no es necesario eliminarla.
Mostramos la nube de puntos entre sqft_living15 y price, viendo así que tiene cierto parecido a una regresión lineal entre ambas.''')
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
        st.pyplot(figr2)
        plt.show()
        return modelo

x = dataframe_regresion['sqft_living15']
y = dataframe_regresion['price']
modelo = plot_regression_model(x,y)
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)
st.write('''Una vez aplicada la regresión lineal, podemos ver que no es del todo precisa, teniendo un coeficiente R2 de 0.27, ya que en el histograma podemos observar ciertas diferencias entre los datos reales y los datos predichos.''')

fig3r = plt.figure(figsize=(20,20))
sns.histplot([np.exp(y_pred), y_test])
plt.title("Histograma de los datos reales y los datos predichos")
st.pyplot(fig3r)
"""El modelo de regresión lineal que hemos aplicado no es del todo bueno, ya que a la hora de predecir el precio de las viviendas, al aplicarle una regresión lineal a los datos, lo que más se asemeja a una recta son los metros cuadrados de la vivienda, pero para poder hacer una predicción más precisa, deberíamos de tener en cuenta más variables, como el número de habitaciones, el número de baños, la zona, la antigüedad, etc.
Por ello, el modelo es capaz de predecir el precio de las viviendas, pero no de forma muy precisa, ya que la puntuación R2 es muy bajo, y el error es muy alto."""