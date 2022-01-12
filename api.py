from pandas.io.formats.format import CategoricalFormatter
from joblib import load 
from flask import Flask, render_template, jsonify, request
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import sys
import os
import pickle
import shutil

## Initialisation ##
app = Flask(__name__)

## Chargement des données qui vont nous servir pour le test :
x_test = pd.read_csv("client_information.csv", index_col='SK_ID_CURR', encoding='utf-8' )

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
    pickle_in = open('/Users/vincentMalfroy/Documents/GitHub/projet_7/LGBM.pkl', 'rb')
    clf = pickle.load(pickle_in)
    score = clf.predict_proba(x_test[x_test.index == int(id_client)]) [:, 1]
    predict = clf.predict(x_test[x_test.index == int(id_client)])
        # Transformation du score en pourcentage avec un arrondissement à 2 chiffres après la virgule 
    score_pourcentage = score*100
    score_pourcentage = np.round(score_pourcentage, 2)
    # creation d'un object JSON
    output = {'prediction': int(predict), 'Risque client en %': float(score_pourcentage)}

    print('Nouvelle Prédiction : \n', output)

    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
