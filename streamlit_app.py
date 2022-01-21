from ssl import Options
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import shap
from urllib.request import urlopen
import json
import pickle 
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from zipfile import ZipFile



## Agrandissement de la page ##
st.set_page_config(layout='wide')

## Importations des bases de données 
zip = ZipFile('client_information.zip')
df = pd.read_csv(zip.open('client_information.csv'), encoding='utf-8').set_index('SK_ID_CURR')

zip = ZipFile('X_dash.zip') 
data = pd.read_csv(zip.open("X_dash.csv"), encoding='utf-8' ).set_index('SK_ID_CURR')

zip = ZipFile('client_information_sample.zip') 
sample = pd.read_csv(zip.open("client_information_sample.csv"), encoding='utf-8' ).set_index('SK_ID_CURR')

id_client = sample.index.values

@st.cache
def load_model():
    '''loading the trained model'''
    pickle_in = open('LGBM.pkl', 'rb') 
    clf = pickle.load(pickle_in)
       

    return clf


################## SIDEBAR ######################

st.sidebar.image('logo.png')
	
#### Input ###
user_input = st.sidebar.selectbox(
	'Entrez le numéro de client:',id_client)
info_client = data[data.index == user_input].iloc[0]

st.sidebar.write('Exploration des caractéristiques du client')
listes = ['Individuelle','Globale', 'Profils similaires']
selection = st.sidebar.selectbox('Interprétation',['','Individuelle','Globale','Profils similaires'])  


####################  CENTRAL ##########################
header_container = st.container()
with header_container:
	st.markdown("<h1 style='text-align: center;color: white'>Simulation de prêt</h1>", unsafe_allow_html=True)
	st.markdown("<h3 style='text-align: center;'>Obtenez une réponse imédiate</h3>", unsafe_allow_html=True)


st.write("**Information sur le client n°**",user_input)
	####### Informations sur le client sélectionnée
client_genre = info_client["CODE_GENDER"]
client_age = info_client["YEARS_BIRTH"]
client_age_binned = info_client['YEARS_BINNED']
client_education = info_client["NAME_EDUCATION_TYPE"]
client_revenu = info_client["AMT_INCOME_TOTAL"]
client_emploi = info_client['YEARS_EMPLOYED']
client_contrat = info_client['NAME_CONTRACT_TYPE']
client_enfant = info_client['CNT_CHILDREN']
client_credit = info_client['AMT_CREDIT']
client_relation = info_client['NAME_FAMILY_STATUS']


container = st.container()
all = st.checkbox("Selectionnez tous les critères")
 
if all:
    options = container.multiselect("Selectionnez un ou plusieurs critères:",
         ['Genre', 'Age', 'Niveau scolaire', 'Revenu du client', "Nombre d'années travaillé", 'Type de contrat',
		 "Nombre d'enfant", 'Montant du crédit', 'Statut maritale'],['Genre', 'Age', 'Niveau scolaire', 'Revenu du client', "Nombre d'années travaillé", 'Type de contrat',
		 "Nombre d'enfant", 'Montant du crédit', 'Statut maritale'])
else:
    options =  container.multiselect("Selectionnez un ou plusieurs critères:",
        ['Genre', 'Age', 'Niveau scolaire', 'Revenu du client', "Nombre d'années travaillé", 'Type de contrat', 
		"Nombre d'enfant", 'Montant du crédit', 'Statut maritale'])


col1, col2, col3 = st.columns(3)
with col1:
	if 'Genre' in options:
		st.markdown("<u>Sexe:</u>", unsafe_allow_html=True)
		st.text(client_genre)
with col2:
	if 'Age' in options:
		st.markdown("<u>Age:</u>", unsafe_allow_html=True)
		st.write(int(client_age), "years")
with col3:
	if 'Niveau scolaire' in options:
		st.markdown("<u>Niveau scolaire:</u>", unsafe_allow_html=True)
		st.write(client_education)

col4, col5, col6 = st.columns(3)
with col4:
	if 'Revenu du client' in options:
		st.markdown("<u>Revenu du client:</u>", unsafe_allow_html=True)
		st.write(int(client_revenu),"$")
with col5:
	if "Nombre d'années travaillé" in options:
		st.markdown("<u>Nombre d'années travaillé:</u>", unsafe_allow_html=True)
		st.write(int(client_emploi), "ans")
with col6:
	if 'Type de contrat' in options:
		st.markdown("<u>Type de contrat :</u>", unsafe_allow_html=True)
		st.write(client_contrat)
col7, col8, col9 = st.columns(3)
with col7:
	if "Nombre d'enfant" in options:
		st.markdown("<u>Nombre d'enfant :</u>", unsafe_allow_html=True)
		st.write(int(client_enfant))
with col8:
	if 'Montant du crédit' in options:
		st.markdown("<u>Montant du crédit :</u>", unsafe_allow_html=True)
		st.write(int(client_credit),"$")
with col9:
	if 'Statut maritale' in options:
		st.markdown("<u>Statut marital :</u>", unsafe_allow_html=True)
		st.write(client_relation)


	### Validation du pret
st.markdown("# Validation du prêt")
st.write(' Seuil de solvabilité avant defaut de paiment (réglé à 50%)') 

# Appel de l'API :
API_url = "https://api-banque-pret.herokuapp.com/credit/" + str(user_input)
with st.spinner('Chargement du score client...'):
	json_url = urlopen(API_url) 

	API_data = json.loads(json_url.read())
	prediction = API_data['risque_client']
	st.write("**Risque de défaut client : **{:.0f} %".format(round(float(prediction), 2)))

#Compute decision according to the best threshold 50% (it's just a guess)
loanResult = 'Status du prêt:' 
if prediction <= 50.0 :
	loanResult += " Validé !"
	st.success(loanResult)
else :
	loanResult += " Refusé..."
	st.error(loanResult)



if selection == 'Individuelle':
	st.markdown("<h2 style='text-align: center;'>Principaux indicateurs influençants le taux de risque</h2>", unsafe_allow_html=True)
	shap.initjs()
	X = sample[sample.index == user_input]

	fig, ax = plt.subplots()
	explainer = shap.TreeExplainer(load_model())
	shap_values = explainer.shap_values(X)
	shap.summary_plot(shap_values, X, max_display=15, plot_size=(5,5))
	st.pyplot(fig)

	fig, ax = plt.subplots(figsize=(5,5))
	explainer = shap.TreeExplainer(load_model())
	shap_values = explainer.shap_values(X)
	shap.summary_plot(shap_values[0], X, max_display=15)
	st.pyplot(fig)

	


if selection == 'Globale':
	st.markdown("<h2 style='text-align: center;'>Principaux indicateurs influençants le taux de risque de la classe 0</h2>", unsafe_allow_html=True)
	st.image('le taux de risque_mean.png')

	st.markdown("<h2 style='text-align: center;'>Principaux indicateurs influençants le taux de risque de la classe 0</h2>", unsafe_allow_html=True)
	st.image('le taux de risque.png')







if selection == 'Profils similaires':
	st.markdown("<h2 style='text-align: center;'>Comparaison des caractéristiques avec un echantillon de client au profils similaires</h2>", unsafe_allow_html=True)	
	
	#PROCHE VOISIN
	contrat_v=data[data['NAME_CONTRACT_TYPE']==client_contrat]
	genre_v=contrat_v[contrat_v['CODE_GENDER']==client_genre]
	education_v=genre_v[genre_v['NAME_EDUCATION_TYPE']==client_education]
	age_v=education_v[education_v['YEARS_BINNED']==client_age_binned]

	if len(age_v) <15:
		set_client_voisin=age_v.sample(len(age_v), random_state=42)
	if len(age_v) >= 15:
		set_client_voisin=age_v.sample(15,random_state=42)
	
	
	st.write('En fonction des caractéristiques socio-demographiques similaire (age, genre, education, type de contrat)')
	

	df_voisin = set_client_voisin.reindex(columns=['EXT_SOURCE_3','EXT_SOURCE_2', 'EXT_SOURCE_1', 'CREDIT_TERM',
                                    'AMT_GOODS_PRICE', 'AMT_CREDIT',
                                    'YEARS_BIRTH', 'AMT_INCOME_TOTAL'])
	
	df_voisin = pd.DataFrame(df_voisin.mean(), columns=['Moyenne_profil_similaire (N=15)'])

	df_client = (data.loc[data.index == user_input])
	df_client = df_client.reindex(columns=['EXT_SOURCE_3','EXT_SOURCE_2', 'EXT_SOURCE_1','CREDIT_TERM',
                                    'AMT_GOODS_PRICE', 'AMT_CREDIT',
                                    'YEARS_BIRTH', 'AMT_INCOME_TOTAL'])
	df_client = df_client.T

	df_client = df_voisin.join(df_client)
	df_client = df_client.style.format('{:.2f}')

	st.table(df_client)
	
	
