"""
import requests
import json
import datetime
import os
"""

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB,CategoricalNB
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.layers import Input, Dense #Pour instancier une couche Dense et une d'Input
from tensorflow.keras.models import Model

import spacy
from nltk.tokenize import word_tokenize

from joblib import dump
"""

import sys
sys.path.append('/app/clean_functions')
from my_models import GBC_predict_df, SVM_predict_df, ANN_predict_df

# initialisation du DAG avec un lancement des task toutes les minutes -----------------------------
"""
Définition du DAG
"""
my_dag = DAG(
    dag_id='airflow_projet',
    description='DAG for sentiment analysis',
    doc_md="""
    ## DAG - AIRFLOW FOR SENTIMENT ANALYSIS
    ---
    Target: predicting sentiment (negative or positive)
    ---
    ### Methodology:
    * Collecting data on the internal database (every 10 minutes)
    * Train 3 different models (GBC, SVM & ANN after wikipedia preprocessing)
    * Save the models
    """,
    tags = ['projet', 'mlops'],
    schedule_interval=None,    #'* * * * *',   # 10 minutes
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False)


# écriture dans le fichier à sauvegarder
with open("/app/clean_data/review_trust_fr_lemmantiser_word+2_VF.csv",'r') as old_file:
    df_old = pd.read_csv(old_file,index_col = 0)
    
def print_test():
    print("df ligne 1 = ",df_old.iloc[0:1,:])  # test pour voir dans le log du task si tout ok
    

#####################################################
#   1. MODELE GBC - GRADIENT BOOSTING CLASSIFIER    #
#####################################################
def GBC_predict():
    GBC_predict_df(# Chemin de stockage des données nettoyées
                   path_data='/app/clean_data/data_MAJ.csv',
                   # Chemin du fichier contenant les stopwords
                   path_stopwords='/app/clean_data/liste_no-stop-words_tokens_unique.xlsx',
                   # Boolean indiquant si on sauve le modèle ou pas
                   save_model=True,
                   # Chemin de sauvegarde du modèle GBC
                   path_save_model='/app/clean_model/GBC_2-updated.pickle',
                   # Boolean indiquant si on sauve le vectorizer ou pas
                   save_vectorizer=True,
                   # Chemin de stockage du vectorizer
                   path_save_vectorizer='/app/clean_model/vectoriser_GBC_2-updated',
                   # Affiche les rapports de performance si True
                   print_report=True
    )



#####################################################
#    2.2 MODELE SVM - SUPPORT VECTOR MACHINES       #
#####################################################
def SVM_predict():
    SVM_predict_df(# Chemin de stockage des données nettoyées
                   path_data='/app/clean_data/review_trust_fr_lemmantiser_word+2_VF.csv',
                   # Chemin du modèle Wikipedia
                   path_model_wiki='/app/clean_model/trained.pickle',
                   # Boolean indiquant si on sauve le modèle ou pas
                   save_model=True,
                   # Chemin de sauvegarde du modèle SVM
                   path_save_model='/app/clean_model/SVM-updated.pickle',
                   # Affiche les rapports de performance si True
                   print_report=True
    )


#####################################################
#    2.3 MODELE ANN ARTIFICEL NEURONAL NETWORK      #
#####################################################
def ANN_predict():
    ANN_predict_df(# Chemin de stockage des données nettoyées
                   path_data='/app/clean_data/review_trust_fr_lemmantiser_word+2_VF.csv',
                   # Chemin du modèle Wikipedia
                   path_model_wiki='/app/clean_model/trained.pickle',
                   # Boolean indiquant si on sauve le modèle ou pas
                   save_model=True,
                   # Chemin de sauvegarde du modèle ANN
                   path_save_model='/app/clean_model/ANN-updated.h5',
                   # Affiche les rapports de performance si True
                   print_report=True
    )


    
# task 1 ------------------------------------------------------------------------------------------
task1 = PythonOperator(
    task_id='collect_data',
    doc_md="""
    ## Collect data and update file '/app/clean_data/data_MAJ.csv'
    * Input: '/app/clean_data/data_MAJ.csv'
    * Output: '/app/clean_data/data_MAJ.csv'
    """,
    python_callable=print_test,
    dag=my_dag)


# task 2 ------------------------------------------------------------------------------------------
task2 = PythonOperator(
    task_id='preprocess_train_save_GBC',
    doc_md="""
    ## Prepocessing, Training, and Saving of the GBC model
    * Input: '/app/clean_data/data_MAJ.csv'
    * Output1: '/app/clean_model/GBC_2-updated.pickle'
    * Output2: '/app/clean_model/vectoriser_GBC_2-updated'
    """,
    python_callable=GBC_predict,
    dag=my_dag)


# task 3 ------------------------------------------------------------------------------------------
task3 = PythonOperator(
    task_id='preprocess_SVM_ANN',
    doc_md="""
    ## Prepocessing for SVM and ANN models
    * Input: '/airflow/clean_data/data_MAJ.csv'
    * Output: '../clean_data/review_trust_fr_lemmantiser_word+2_VF.csv'
    """,
    python_callable=print_test,
    dag=my_dag)


# task 3_1 ------------------------------------------------------------------------------------------
task3_1 = PythonOperator(
    task_id='train_save_SVM',
    doc_md="""
    ## Training and Saving of the SVM model
    * Input: '../clean_data/review_trust_fr_lemmantiser_word+2_VF.csv'
    * Output: '../clean_model/SVM-updated.pickle'
    """,
    python_callable=SVM_predict,
    dag=my_dag)


# task 3_2 ------------------------------------------------------------------------------------------
task3_2 = PythonOperator(
    task_id='train_save_ANN',
    doc_md="""
    ## Training and Saving of the ANN model
    * Input: '../clean_data/review_trust_fr_lemmantiser_word+2_VF.csv'
    * Output: '../clean_model/ANN-updated.h5'
    """,
    python_callable=ANN_predict,
    dag=my_dag)


# Enchainement des taches
task1 >> [task2, task3]
task3 >> [task3_1, task3_2]