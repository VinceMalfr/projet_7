import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly
import shap

st.set_page_config(layout='wide')

## Importations des bases de données 
df = pd.read_csv('df_final.csv')


################## SIDEBAR ######################

st.sidebar.image('logo.png')
	
#### Input ###
user_input = st.sidebar.selectbox(
	'Entrez le numéro de client:',df['SK_ID_CURR'].tolist())
info_client = df[df["SK_ID_CURR"]==user_input].iloc[0]

st.sidebar.write('Exploration des caractéristiques du client')
listes = ['Individuelle','Globale', 'Profils similaires']
multi_select=st.sidebar.multiselect('Interprétation',listes, key='listes')




####################  CENTRAL ##########################
header_container = st.container()
with header_container:
	#image= Image.open('logo_central.png')
	#st.image(image,use_column_width=True)
	st.markdown("<h1 style='text-align: center;color: white'>Simulation de prêt</h1>", unsafe_allow_html=True)
	st.markdown("<h3 style='text-align: center;'>Obtenez une réponse imédiate</h3>", unsafe_allow_html=True)
	

st.write("**Information sur le client n°**",user_input)
	####### Informations sur le client sélectionnée
client_genre = info_client["GENDER"]
client_age = info_client["YEARS_BIRTH"]
client_education = info_client["NAME_EDUCATION_TYPE"]
client_revenu = info_client["AMT_INCOME_TOTAL_x"]
client_emploi = info_client['YEARS_EMPLOYED']

col1, col2 = st.columns(2)
with col1:
	## Genre
	st.markdown("<u>Sexe:</u>", unsafe_allow_html=True)
	st.text(client_genre)
with col2:
	## Age
	st.markdown("<u>Age:</u>", unsafe_allow_html=True)
	st.write(int(client_age), "years")

check = st.checkbox("Plus d'informations")
if check :
	col3, col4, col5 = st.columns(3)
	with col3:
	## Education
		st.markdown("<u>Niveau scolaire:</u>", unsafe_allow_html=True)
		st.write(client_education)
	with col4:
		## revenu
		st.markdown("<u>Revenu du client:</u>", unsafe_allow_html=True)
		st.write(int(client_revenu), "$")
	with col5:
		## Nombre d'années employé
		st.markdown("<u>Nomre d'années travaillé:</u>", unsafe_allow_html=True)
		st.write(int(client_emploi))
	

	### Validation du pret
st.markdown("# Validation du prêt")
st.write(' Seuil de solvabilité avant defaut de paiment (réglé à 50%)') 
st.write("Le score obtenu est de :", (info_client['TARGET']*100).round(0),'%')
loanResult = 'Status du prêt:' 
if info_client['TARGET'] <= 0.45 :
	loanResult += " Validé !"
	st.success(loanResult)
else :
	loanResult += " Refusé..."
	st.error(loanResult)
	
