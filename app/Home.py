import streamlit as st

st.set_page_config(
    page_title="💄Final Project",
)

st.title("😈 Proyecto Final")

dict = {
    "Javier Míguelez": "https://github.com/Xavitheforce",
    "Juan Medina": "https://github.com/jmedina28",
    "Diego de Santos": "https://github.com/Diegodesantos1"
}
st.write('''👁️👄👁️
    Tras la realización del curso de Data Science, nos hemos dispuesto a realizar un proyecto final.
    El proyecto consiste en la creación de una aplicación web que muestre los resultados de dos modelos, uno de regresión y otro de clasificación.
    Los autores de este proyecto son los siguientes:''')

for key, value in dict.items():
    st.write(f"👉 {key} 🥶 {value}")