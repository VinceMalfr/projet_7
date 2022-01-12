import streamlit as st
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly
import shap
from urllib.request import urlopen
import json
import pickle 


st.set_page_config(layout='wide')

## Importations des bases de données 
#df = pd.read_csv('client_information_sample.csv', index_col='SK_ID_CURR', encoding='utf-8')
data = pd.read_csv("X_dash.csv", index_col='SK_ID_CURR', encoding='utf-8' )
sample = pd.read_csv("client_information_sample.csv", index_col='SK_ID_CURR', encoding='utf-8' )

id_client = sample.index.values

def credit_factors(model, patient,data):

    explainer = shap.KernelExplainer(model, data)
    shap_values = explainer.shap_values(patient)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)

def load_model():
    '''loading the trained model'''
    pickle_in = open('/Users/vincentMalfroy/Documents/GitHub/projet_7/LGBM.pkl', 'rb') 
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
	#image= Image.open('logo_central.png')
	#st.image(image,use_column_width=True)
	st.markdown("<h1 style='text-align: center;color: white'>Simulation de prêt</h1>", unsafe_allow_html=True)
	st.markdown("<h3 style='text-align: center;'>Obtenez une réponse imédiate</h3>", unsafe_allow_html=True)
	

st.write("**Information sur le client n°**",user_input)
	####### Informations sur le client sélectionnée
client_genre = info_client["CODE_GENDER"]
client_age = info_client["YEARS_BIRTH"]
client_education = info_client["NAME_EDUCATION_TYPE"]
client_revenu = info_client["AMT_INCOME_TOTAL"]
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

	###Figure 
	col3, col4, col5 = st.columns(3)
	with col3:
		### Age du client dans la population totale 
		fig, ax = plt.subplots(figsize=(10,5))
		sns.histplot(data= data, x="YEARS_BIRTH", bins=10)
		ax.axvline(int(info_client["YEARS_BIRTH"]), color='green', linestyle='--')
		ax.set(title='Age du client dans la population totale', xlabel='Age', ylabel='')
		st.pyplot(fig)

	with col4:
		### Revenu du client dans la population totale 
		fig, ax = plt.subplots(figsize=(10,5))
		sns.histplot(data= data, x="AMT_INCOME_TOTAL", bins=10)
		ax.axvline(int(info_client["AMT_INCOME_TOTAL"]), color='green', linestyle='--')
		ax.set(title='Revenu du client dans la population totale ', xlabel='revenu', ylabel='')
		st.pyplot(fig)

	with col5:

		### Nombre d'années travaillé dans la population totale
		fig, ax = plt.subplots(figsize=(10,5))
		sns.histplot(data= data, x="YEARS_EMPLOYED", bins=30)
		ax.axvline(int(info_client["YEARS_EMPLOYED"]), color='green', linestyle='--')
		ax.set(title="Nombre d'années travaillé dans la population totale", xlabel='Année travaillés', ylabel='')
		st.pyplot(fig)


	### Validation du pret
st.markdown("# Validation du prêt")
st.write(' Seuil de solvabilité avant defaut de paiment (réglé à 50%)') 


# Appel de l'API :
API_url = "http://localhost:5000/credit/" + str(user_input) 

with st.spinner('Chargement du score client...'):
	json_url = urlopen(API_url) 

	API_data = json.loads(json_url.read())
	prediction = API_data['Risque client en %']
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
	#X = sample.iloc[:, :-1]
	X = sample[sample.index == user_input]

	fig, ax = plt.subplots(figsize=(10, 10))
	explainer = shap.TreeExplainer(load_model())
	shap_values = explainer.shap_values(X)
	shap.summary_plot(shap_values[0], X, plot_type ="bar", color_bar=True, plot_size=(5, 5))
	st.pyplot(fig)



explainer = shap.TreeExplainer(load_model())
shap_values = explainer.shap_values(X)
shap.initjs()
test = shap.force_plot(explainer.expected_value[1], shap_values[1], X)
st.pyplot(test)




