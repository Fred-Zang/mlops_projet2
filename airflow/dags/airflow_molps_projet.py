import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


import sys
sys.path.append('/app/clean_functions')   # chemin /airflow/clean_functions error => on passe par le volume /app
from my_models import GBC_predict_df, SVM_predict_df, ANN_predict_df
from my_functions import token_lemmatiser

# initialisation du DAG avec un lancement des task toutes les minutes ----
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
    tags=['projet', 'mlops'],
    schedule_interval=None,  # '* * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False)


####################################################################################
#   0. PRE-PREOCESSING DE DATA_MAJ.CSV  => CREATION DE DATA_PREPROCESSING_V1.CSV   #
####################################################################################
def Data_Preprocessing():
    token_lemmatiser(path="/app/clean_data/data_MAJ.csv")


#####################################################
#   1. MODELE GBC - GRADIENT BOOSTING CLASSIFIER    #
#####################################################
def GBC_preprocess_train_predict():
    GBC_predict_df(path_data='/app/clean_data/data_MAJ.csv')


#####################################################
#    2.2 MODELE SVM - SUPPORT VECTOR MACHINES       #
#####################################################
def SVM_predict():
    SVM_predict_df(path_data='/app/clean_data/data_preprocess_v1.csv')


#####################################################
#    2.3 MODELE ANN ARTIFICEL NEURONAL NETWORK      #
#####################################################
def ANN_predict():
    ANN_predict_df(path_data='/app/clean_data/data_preprocess_v1.csv')


task1 = PythonOperator(
    task_id='preprocessing_data_MAJ',
    doc_md="""
    ## Preprocessing de data_MAJ.csv => + columns lemm + word+2
    * Input: data_MAJ.csv = data originale sans colonnes de preprocessing
    * Output: création de data_preprocess_v1.csv dans le dossier local /airflow/clean_data
            avec 3 colonnes supplémentaires lemmatiser,words+2,no_stop_words
    """,
    python_callable=Data_Preprocessing,
    dag=my_dag)


task2 = PythonOperator(
    task_id="preprocess_2_train_save_GBC2_save_classif",
    doc_md="""
    ## Preprocessing & Training & Saving of the GBC2 model
     + calculating and saving Classification_report on Jason file
    * Input: Input: data_MAJ.csv = data originale sans colonnes de preprocessing 
    * Output: '/clean_model/GBC2_sav-DAG.pickle'
    """,
    python_callable=GBC_preprocess_train_predict,
    dag=my_dag)


# task 3_1 ---------------------------------------------------------------
task3_1 = PythonOperator(
    task_id='train_save_SVM_save_classif',
    doc_md="""
    ## Training and Saving of the SVM model
     + calculating and saving Classification_report on Jason file
    * Input: '/clean_data/data_preprocess_v1.csv'
    * Output: '/clean_model/SVM-updated.pickle'
    """,
    python_callable=SVM_predict,
    dag=my_dag)


# task 3_2 ---------------------------------------------------------------
task3_2 = PythonOperator(
    task_id='train_save_ANN_save_classif',
    doc_md="""
    ## Training and Saving of the ANN model.pickle
     + calculating and saving Classification_report on Jason file
    * Input: '/clean_data/'/clean_data/data_preprocess_v1.csv''
    * Output: '/clean_model/ANN-updated.h5'
    """,
    python_callable=ANN_predict,
    dag=my_dag)


# Enchainement des taches
# task1 >> [task2, task3]
# task3 >> [task3_1, task3_2]

task1 >> [task3_1, task3_2]
task2
