# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
import pickle
from PIL import Image
import streamlit as st
# import seaborn as sns
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import webbrowser

import sys
import re
import spacy
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  # pour modele ANN.h5
nltk.download('punkt')                  # pour modele ANN.h5


# from wordcloud import WordCloud


# --------------------chargement des modèles--------------------------#

nlp = spacy.load('fr_core_news_md')
# nlp = pickle.load(open("/airflow/clean_model/nlp_SAV_core_news_md.pickle","rb"))

# load Vectorizer For Gender Prediction                                           /airflow/clean_model/vectoriser_ber
# ---------  chargement des modèles et dataframes diverses ------------- #
GBC_2_vectorizer = pickle.load(
    open("/airflow/clean_model/vectoriser_GBC_2-sav_DAG", "rb"))


# load pre-trained model
trained = pickle.load(open(r"/airflow/clean_model/trained.pickle", "rb"))

# load Model For Prediction
GBC_2_model = pickle.load(
    open("/airflow/clean_model/GBC_2-sav_DAG.pickle", "rb"))

SVM_model = pickle.load(open("/airflow/clean_model/SVM_sav-DAG.pickle", "rb"))


# ---------  Fonction de traitement des corpus ------------- #
@st.cache
def text_tokeniser(my_text):
    nlp = spacy.load('fr_core_news_md')
    docx = nlp(my_text)
    # tokens = [ token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(
        token.text, token.lemma_))for token in docx]
    return allData


def text_tokeniser2(my_text):
    nlp = spacy.load('fr_core_news_md')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]

    return tokens

# ---------  Fonction de traitement des corpus ------------- #
# fonction pour transformer un document ( ici une ligne de commentaire)
# en vecteur à partir des tokens qui le compose
    # entrée : line_com lignes commentaires à traiter
    #          model pré-entrainé
    # sortie : vecteur représentant le document


def my_doc2vec(doc, trained):
    # dimension de représentation
    p = trained.vectors.shape[1]   # p = 100
    # intit du vecteur
    vec = np.zeros(p)  # array (100,)
    # nbre de tokens trouvés
    nb = 0
    # traitement de chaque token de la ligne de commentaire
    for tk in doc:
        # ne traiter que les tokens reconnus
        try:
            values = trained[tk]  # on récupère le vecteur du token concerné
            vec = vec + values   # on incrémente ce vecteur dans vec
            nb = nb + 1.0        # le compteur de token d'incrémente
        except BaseException:
            pass  # pour lever l'erreur si aucun token trouvé dans notre modele pré-entrainé
    # moyenne des valeurs uniquement si on a des tokens reconnus
    if (nb > 0.0):
        vec = vec / nb
    return vec  # renvoie le vecteur moyenné ou un vecteur nul si aucun token trouvé


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

# ---------  Fonction de prédiction à la demande ------------- #


def present_part(choose_do):
    if choose_do == 'Modelisations à la demande':

        st.markdown('# Modelisations à la demande')

        corpus = st.text_area(
            "Entrer un commentaire puis taper Ctrl + Entrée pour le valider",
            "Ecrire ici ..")
        corpus = corpus.lower()

        models = ['GBC_2', 'SVM', 'ANN']
        model = st.selectbox(label='Choix du modèle', options=models)
        # charger le modele préentraine widipedia en avance: je ne considère
        # pas l'opimisation des temps ici. pas encore, on charge le modèle meme
        # si le GBC_2 n'utilse pas

        if model == 'GBC_2':
            pred = GBC_2(corpus)
            sentiment(pred)
        elif model == 'SVM':
            pred = SVM(corpus)
            sentiment(pred)
        elif model == 'ANN':
            # PAS BIEN NOMMEE #######################################
            ANN = tf.keras.models.load_model(
                r'/airflow/clean_model/ANN-sav-DAG.h5')
            pred = prediction(corpus, ANN)
        # st.write(pred)
            pred = pred[0].tolist()
            pred = pred.index(max(pred))
            sentiment(pred)
            st.write(pred)
        else:
            st.write('')
            # je ne sais pas si nous avons besoin d'ajouter autres modèle ou
            # pas


# ---------  Fonction d'affichage de sentime ------------- #
def sentiment(pred):
    if pred == 1:
        st.subheader("Prédiction = Sentiment Positif:smiley: ")
        # st.write('Votre commentaire est classifié: ', pred)
        # st.write('Merci à votre *positif* commentaire :smile:')
    else:
        st.subheader("Prédiction = Sentiment Négatif :angry: ")
        # st.write('Votre commentaire est classifié: ', pred)
        # st.write('Merci à votre *négative* commentaire :angry:. Pour améliorer notre service, la site vous contactera.')

# ---------  Fonction la prédiction par SVM_wiki ------------- #


def SVM(corpus):
    # 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
    trained = pickle.load(open('/airflow/clean_model/trained.pickle', 'rb'))
    # charger le model
    # 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
    SVM_wiki = pickle.load(
        open(
            '/airflow/clean_model/SVM_sav-DAG.pickle',
            'rb'))
    text_vec = my_doc2vec(corpus, trained)
    text_vector = pd.DataFrame(my_doc2vec(corpus, trained)).T
    pred = SVM_wiki.predict(text_vector)
    return int(pred)

# ---------  Fonction la prédiction par GBC ------------- #


def GBC_2(corpus):
    # charger le vecteur
    # 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
    vectorizer = pickle.load(
        open(
            '/airflow/clean_model/vectoriser_GBC_2-sav_DAG',
            'rb'))
    # charger le model
    # 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
    GBC_2 = pickle.load(
        open(
            '/airflow/clean_model/GBC_2-sav_DAG.pickle',
            'rb'))
    text = pd.Series(corpus)
    text_vec = vectorizer.transform(text).todense()
    # st.write('vectorization for GBC2 ',text_vec)
    pred = GBC_2.predict(text_vec)
    pred = int(pred)
    return pred

# ---------  Fonction la prédiction par ANN, GBC_3 et SVM ------------- #


def prediction(text, model):
    # 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
    trained = pickle.load(open('/airflow/clean_model/trained.pickle', 'rb'))
    my_doc = text.lower()
    my_doc_tk = word_tokenize(my_doc)

    def lemms(corpus_tk):
        doc = nlp(" ".join(corpus_tk))
        lemm = [token.lemma_ for token in doc]
        return lemm
    my_doc_lem = lemms(my_doc_tk)

    def stop_words_filtering(mots, stop_words):
        tokens = []
        for mot in mots:
            if mot not in stop_words:
                tokens.append(mot)
        return tokens
    # 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
    df_stop_word_xls = pd.read_excel(
        '/airflow/clean_data/liste_no-stop-words_tokens_unique.xlsx',
        header=None)
    update_list_fr = list(df_stop_word_xls[0])
    # initialisation de la variable des mots vides
    stop_words = set()
    stop_words.update(update_list_fr)
    my_doc_sw = stop_words_filtering(my_doc_lem, stop_words)
    my_vec = my_doc2vec(my_doc_sw, trained)

    pred_my_doc = model.predict(my_vec.reshape(1, -1))
    return pred_my_doc

# ---------------------- créer 4 pages différentes sur notre streamlit --------------------------------------  #
# créer une liste de 4 noms des pages


pages = [
    "Le Projet en DETAILS",
    "Cahier des Charges",
    "Améliorations à prévoir et Problèmatiques",
    "Modélisations à la demande",
    "Conclusion et REMERciements"]

page = st.sidebar.radio("Aller vers", pages)

# ------------------------------------------------------------------------------------------
# séparation des pages
# ------------------------------------------------------------------------------------------

if page == pages[0]:  # sur la page 0 Introduction
    # affichage
    st.write("### Frontend Streamlit")
    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        "/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png")

    # title du page
    st.markdown(
        "<h1 style='text-align: center; color: white;'>SatisPy Projet - version MLOps</h1>",
        unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------
# séparation des pages
# ------------------------------------------------------------------------------------------
elif page == pages[1]:  # sur la page 1 Dataviz
    # affichage
    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        '/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
    st.markdown(
        "<h2 style='text-align: center; color: white;'>Cahier des Charges</h2>",
        unsafe_allow_html=True)
    st.write("Schéma d'implantation - TOUT A FAIRE !")
    st.image("/airflow/data_others/JPG-PNG/implantation.png")
    st.image("/airflow/data_others/JPG-PNG/Capture d’écran 2023-03-23 183210.png")
    st.write("Suivi du cahier des charges  => à refaire proprement + images")
    st.image("/airflow/data_others/JPG-PNG/cahier_des_charges_1.png")
    st.image("/airflow/data_others/JPG-PNG/cahier_des_charges_2.png")
    st.image("/airflow/data_others/JPG-PNG/cahier_des_charges_3.png")


# ------------------------------------------------------------------------------------------
# sépration des pages
# ------------------------------------------------------------------------------------------
# sur la page 2
elif page == pages[2]:
    # affichage
    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        '/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
    # title du page
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Améliorations à prévoir et Problèmatiques</h1>",
        unsafe_allow_html=True)

    st.write("VULNÉRABILITÉ DES CONTAINERS")
    st.markdown(""" - exemple ici vulnératbilité sur python 3.9""")
    st.image("/airflow/data_others/JPG-PNG/vulnerability_containers.png")

    st.write("POIDS DES IMAGES BUILDÉES")
    st.markdown(""" - Les images contruites pour STreamlit, FastAPI et AirFlow sont assez lourde avec respectivement 1 GB, 2GB et 2GB""")
    st.image("/airflow/data_others/JPG-PNG/Poids_des_images.png")
    st.write('')
    st.write("VERSIONNING DES PACKAGES")
    st.image("/airflow/data_others/JPG-PNG/requirement_streamlit.png")
    st.write('')
    st.write("USAGE DE RAM IMPORTANTE")
    st.markdown(
        """ - problématiques de chargement en local et une utilisation de RAM trop importante si RAM pc < 16 Go """)
    st.image("/airflow/data_others/JPG-PNG/Vmmem.png")
    st.image("/airflow/data_others/JPG-PNG/Vue_processeur.png")
    st.write('')

# ------------------------------------------------------------------------------------------
# sépration des pages
# ------------------------------------------------------------------------------------------
elif page == pages[3]:  # sur la page 3 Modelisation

    # TOUS LES MODELES ET DATA ICI SONT RECUPERES DANS /airflow/c

    # affichage
    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        '/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')

    st.markdown(
        "<h2 style='text-align: center; color: white;'>Nos 3 Modèles présentés ici</h2>",
        unsafe_allow_html=True)

    # title du page
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Modélisations à la demande</h1>",
        unsafe_allow_html=True)

    # st.write("# Modélisations à la demande")
    message = st.text_area(
        "Entrer un commentaire puis taper Ctrl + Entrée pour le valider",
        "Ecrire ici ..")
    corpus = message.lower()
    if st.button("Gradient Boosting Classifier 2"):
        pred = GBC_2(corpus)
# 🛑🛑mélange anglais et français ??🛑🛑
        st.success('Your message was classified as {}'.format(pred))
        sentiment = sentiment(pred)
        st.write(
            "Ce modèle atteint une précision de 89% sur les 2 sentiments et donc malgré tout un taux d'erreurs de 11%")

    if st.button(
            "pré-entrainement avec Wikipedia2vec puis modélisation par SVM"):
        pred = SVM(corpus)
# 🛑🛑mélange anglais et français ??🛑🛑
        st.success('Your message was classified as {}'.format(pred))
        sentiment(pred)
        st.write(
            "Ce modèle atteint une précision de 90% sur les 'satisfaits' et seulement 76% sur les 'mécontents")

    if st.button("ANN Réseaux de Neurones Articiciels"):
        ANN = tf.keras.models.load_model('/airflow/clean_model/ANN-sav-DAG.h5')
        pred = prediction(corpus, ANN)
        st.write(pred)
        pred = pred[0].tolist()
        pred = pred.index(max(pred))
        sentiment(pred)
        st.write('')
    st.write("Ce modèle atteint une précision de 86% sur les 'satisfaits' et seulement 73% sur les 'mécontents")
# 🛑🛑ajouter les image dans le dossier et décommenté🛑🛑
    # st.image("sample-image.jpg")  # "ANN-layers_fit_confusion_classifreport.jpg")
    # #st.write('add img')
    # st.write("La fonction Loss (coût des erreurs à minimiser) termine à 29.9% ce qui n'est pas optimum ")

# ------------------------------------------------------------------------------------------
# sépration des pages
# ------------------------------------------------------------------------------------------

elif page == pages[4]:  # sur la page 4 Conclusion
    # affichage
    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        '/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')

    # title du page
    st.markdown(
        "<h1 style='text-align: center; color: white;'>✨ Conclusion ✨</h1>",
        unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.write(
            "Traiter un sujet comme celui-ci a été passionnant et nous aimerions le poursuivre encore")
        st.write("")
        st.write("Nous y avons mis beaucoup de cœur, à découvrir toutes ces notions à travers nos modules de cours et à dérouler une grande quantité de modèles pour trouver des approches et ressources différentes.")
        st.write("")
        st.write("Nous avons le sentiment d’en être qu’au tout début de notre étude, tant il nous reste de questions à résoudre, de tests à réaliser et de nouvelles pistes à explorer.")
        st.write("")
        st.write("Nous espérons que sa lecture vous a été agréable et vous recommandons de parcourir les vidéos et liens donnés en annexe si vous souhaitez en savoir plus sur le sujet.")
        st.write("")
        st.markdown(
            "<h1 style='text-align: center; color: white;'>🌼​ Remerciements 🌼​</h1>",
            unsafe_allow_html=True)
        st.write("")
        st.write("Nous tenons à remercier toute l’équipe de DataScientest pour son écoute et leurs conseils, et particulièrement notre mentor de projet Antoine qui nous a suivi chaque semaine en réunion zoom avec un sourire et une patience admirable.")
