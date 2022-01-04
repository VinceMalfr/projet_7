import streamlit as st
import pandas as pd

st.write("""
	# My first app
	Hello *world!*
	""")
st.title("Streamlit sliders")
st subheader("Slider 1:")
x = st.slider('a number between 0-100')

st.write("Slider number:", x)
	
