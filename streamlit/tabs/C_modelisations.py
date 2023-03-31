import streamlit as st
import pandas as pd
import numpy as np

from my_functions_streamlit import GBC_2, sentiment, SVM, prediction
from tensorflow.keras.models import load_model

title = "Modélisations à la demande et surveillance des métriques"
sidebar_name = "Modélisations à la demande et surveillance des métriques"


def run():

    st.title(title)

    st.markdown("---")

    # TOUS LES MODELES ET DATA ICI SONT RECUPERES DANS /airflow/clean_data et /airflow/clean_model

    message = st.text_area(
        "Entrer un commentaire puis taper Ctrl + Entrée pour le valider",
        "Ecrire ici ..")
    corpus = message.lower()


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
        st.write(pred)
        pred = pred[0].tolist()
        pred = pred.index(max(pred))
        sentiment(pred)
        st.write('')

        

