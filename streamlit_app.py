import streamlit as st
import pandas as pd

st.write("""
	# My first app
	Hello *world!*
	""")
st.title("Simulation de prêt")
st.subheader("Obtenez une réponse imédiate:")
x = st.slider('a number between 0-100')

st.write("Slider number:", x)
	
