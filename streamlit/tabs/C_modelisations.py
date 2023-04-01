import streamlit as st
import pandas as pd
import numpy as np

from my_functions_streamlit import GBC_2, sentiment, SVM, prediction
from tensorflow.keras.models import load_model

title = "Modélisations à la demande"
sidebar_name = "Modélisations à la demande"


def run():

    st.title(title)

    st.markdown("---")

    # TOUS LES MODELES ET DATA ICI SONT RECUPERES DANS /airflow/clean_data et
    # /airflow/clean_model
    st.write(
        "#### :green[Entrez un Commentaire] puis faire :orange[CTRL + ENTER] pour le valider")
    message = st.text_area(
        " ",
        "Ecrire ici ..")
    corpus = message.lower()

    st.write(
        "#### :green[Choisissez un Modèle] pour lancer la :orange[prédiction]")

    if st.button("Gradient Boosting Classifier 2"):
        pred = GBC_2(corpus)
        st.success('Your message was classified as {}'.format(pred))
        sentiment(pred)
        st.write("")

    if st.button(
            "pré-entrainement avec Wikipedia2vec puis modélisation par SVM"):
        pred = SVM(corpus)
        st.success('Your message was classified as {}'.format(pred))
        sentiment(pred)
        st.write("")

    if st.button("ANN Réseaux de Neurones Articiciels"):
        ANN = load_model('/airflow/clean_model/ANN-sav-DAG.h5')
        pred = prediction(corpus, ANN)
        st.success('calculated probability')
        st.write(pred)
        pred = pred[0].tolist()
        pred = pred.index(max(pred))
        sentiment(pred)
        st.write('')
