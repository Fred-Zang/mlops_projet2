import requests
import json
import datetime
import os
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

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



# initialisation du DAG avec un lancement des task toutes les minutes -----------------------------
my_dag = DAG(
    dag_id='airflow_projet',
    tags = ['projet', 'mlops'],
    schedule_interval=None,
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
    
    
# task 1 ------------------------------------------------------------------------------------------
task1 = PythonOperator(
    task_id='task1',
    python_callable=print_test,
    dag=my_dag)






