
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser

import sys
import re
import spacy
import gensim
from nltk.corpus import stopwords

from PIL import Image

import pickle
import joblib
#from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# --------------------chargement des modèles--------------------------#
# load Vectorizer For Gender Prediction                                           data/Modeles/vectoriser_ber
# ---------  chargement des modèles et dataframes diverses ------------- #
GBC_2_vectorizer = pickle.load(open("/data/Modeles/vectoriser_GBC_2","rb"))
Bernoulli_vectorizer = pickle.load(open("/data/Modeles/vectoriser_ber","rb"))

# load pre-trained model
trained = pickle.load(open(r"/data/Modeles/trained.pickle","rb"))

# load Model For Prediction
GBC_2_model = pickle.load(open("/data/Modeles/GBC_2.pickle","rb"))
Bernoulli_model = pickle.load(open("/data/Modeles/ber.pickle","rb"))
SVM_model = pickle.load(open("/data/Modeles/SVM.pickle","rb"))

# ---------  Fonction d'affichage ------------- #


def insert_img(img):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(img)
    with col3:
        st.write(' ')


def insert_head(img1, img2):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(img1)
    with col2:
        st.write(' ')
    with col3:
        st.write(' ')
    with col4:
        st.write(' ')
    with col5:
        st.image(img2)


# ---------------------- créer 4 pages différentes sur notre streamlit --------------------------------------  #
# créer une liste de 4 noms des pages

pages = [
    "Le Projet en DETAILS",
    "Dataviz du Projet",
    "Filtrages, Tokenisations, Lemmatisations et Vectorisations",
    "Modélisations à la demande",
    "Conclusion et REMERciements"]

page = st.sidebar.radio("Aller vers", pages)

# ------------------------------------------------------------------------------------------
# sépration des pages
# ------------------------------------------------------------------------------------------

if page == pages[0]:  # sur la page 0 Introduction
    # affichage
    st.write("### Frontend Streamlit")
    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        "/data/logo_SatisPy_Project.png")

    # title du page
    st.markdown(
        "<h1 style='text-align: center; color: white;'>SatisPy Projet</h1>",
        unsafe_allow_html=True)
    # écrire du texte ( # = taille, ##, ###) ici le titre de la page
    st.write("### Préambule QUE DES BULLES !!!:")
    st.write("Tout comme le rapport de projet, cette présentation Streamlit est destinée à tout public et ne comporte aucune ligne de code.😉​")
    st.write("### Rapide aperçu du Projet :")
    st.write(
        "On nous a remis un dataset 'reviews_trust.csv' comportant 19.863 lignes et 11 colonnes qui correspond aux commentaires clients \
        et notation de 1 à 5 étoiles sur leurs achats de produit sur 2 sites marchands 'ShowRoom' et 'VeePee'. Ces commentaires proviennent de \
            2 sources récoltant les avis, 'TrustedShop' et 'TrustPilot' et voici un extrait du dataset :")
    st.write("---est ce que ça marche ? ---")
