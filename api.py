from flask import Flask, render_template, jsonify, request
import json
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm as lgb
import pickle
import cloudpickle
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
## Initialisation ##
app = Flask(__name__)

## Chargement des données qui vont nous servir pour le test :
x_test = pd.read_csv("client_information.csv", index_col='SK_ID_CURR', encoding='utf-8' )

## Chargement de notre modèle LGBM
pickle_in = open('/API_heroku/LGBM.pkl', 'rb')
clf = pickle.load(pickle_in)

#Routage de app
@app.route("/")
def hello():
    """
    ping the API
    """

    return jsonify({"text": "Hello, l'API est en ligne"})



# On importe le modèle déja entrainé grace à pickle 
# La route est constitué de credit puis du noms d'un des clients ex:'100002'
@app.route('/credit/<id_client>', methods=["GET"])
def credit(id_client): 
    score = clf.predict_proba(x_test[x_test.index == int(id_client)]) [:, 1]
    predict = clf.predict(x_test[x_test.index == int(id_client)])
        # Transformation du score en pourcentage avec un arrondissement à 2 chiffres après la virgule 
    score_pourcentage = score*100
    score_pourcentage = np.round(score_pourcentage, 2)
    # creation d'un object JSON
    output = {'prediction': int(predict), 'risque_client': float(score_pourcentage)}

    print('Nouvelle Prédiction : \n', output)

    return jsonify(output)
# mettre le modèle a l'ext
# fichier requiement.text 
# gunicorn permet de lancer l'application 


if __name__ == '__main__':
    app.run(debug=True)