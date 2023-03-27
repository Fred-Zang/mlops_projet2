import streamlit as st
import pandas as pd
import pickle
import spacy
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import sys
sys.path.append('/airflow/clean_functions')
from my_functions import my_doc2vec

nlp = spacy.load('fr_core_news_md')


# ---------  Fonction d'affichage des logos DataScientest et SatisPy ------------- #
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



# ---------  Fonction la prédiction par GBC ------------- #
def GBC_2(corpus):
    # charger le vecteur
    vectorizer = pickle.load(
        open(
            '/airflow/clean_model/vectoriser_GBC_2-sav_DAG',
            'rb'))
    # charger le model
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



# ---------  Fonction la prédiction par SVM_wiki ------------- #
def SVM(corpus):
    trained = pickle.load(open('/airflow/clean_model/trained.pickle', 'rb'))
    # charger le model
    SVM_wiki = pickle.load(
        open(
            '/airflow/clean_model/SVM_sav-DAG.pickle',
            'rb'))
    #text_vec = my_doc2vec(corpus, trained)
    text_vector = pd.DataFrame(my_doc2vec(corpus, trained)).T
    pred = SVM_wiki.predict(text_vector)
    return int(pred)



# ---------  Fonction la prédiction par ANN ------------- #
def prediction(text, model):
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