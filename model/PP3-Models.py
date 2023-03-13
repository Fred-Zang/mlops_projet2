#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://datascientest.fr/train/assets/logo_datascientest.png", style="height:150px"/></center>
# 
# <hr style="border-width:2px;border-color:#75DFC1">
# 
# <center><img src="https://assets-datascientest.s3.eu-west-1.amazonaws.com/logo+SatisPy+Project.png", style="height:150px"/></center>
# 
# 
# <center><h1> Modèles - Analyse & Tableaux d'erreurs </h1></center>

# # Objectifs :
# 
# L'objectif est de pouvoir lancer plusieurs modélisations sur notre data_set original, ceux filtrés ou sur d'autres variables préparées
# - Utiliser des classifier de bases du Machine Learning
# - Utiliser des modèles pré-entrainés pour y ajouter nos modèles prédictifs
# - Utiliser nos bases acquises en Deep Learning pour modéliser des réseaux de neurones
# - Comparer les scores
# - Comparer les erreurs dans un tableau spécifiques
# - lancer des prédictions rapides à la volée

# chargement des packages
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from time import time
import pickle
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB,CategoricalNB
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout #Pour instancier une couche Dense et une d'Input
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
import spacy

# Versions
print('Version des librairies utilisées :')
print('Python                : ' + sys.version)
print('NumPy                 : ' + np.version.full_version)
print('Pandas                : ' + pd.__version__)
print('Matplotlib.pyplot     : ' + matplotlib.__version__)
print('Seaborn               : ' + sns.__version__)
print('NaturaLanguageTooKit  : ' + nltk.__version__)
print('sklearn               : ' + sklearn.__version__)
#print('Tensorflow            : ' + tensorflow.__version__)
#print('Spacy                 : ' + spacy.__version__)


# préparation de l'affichage des dataframes => même fonction que le notebook data_explo à séparer dans un outils.py à importer
dico_pd_option = {
    'display.max_rows': 100,     # nbre max de lignes 
    'display.max_column': 50,    # nbre max de colonnes
    'display.width': 50,         # largeur lignes 
    'display.precision': 2,      # precision des valeurs
    'display.max_colwidth': 100  # largeurs colonnes
}
for cle, val in dico_pd_option.items():
    pd.set_option(cle, val)
    
pd.options.display.float_format = "{:,.3f}".format  # affichage pd float à 3 décimales


"""
# # 4- Modélisation des données
# 
# Il y a 3 modèles d'apprentissage retenus pour le projet. Les données d'entrées sont différentes. 
# 
# Pour faire la modélisation, les étapes principales : 
# * 1- Préparation des données
# * 2- Construction du modèle
# * 3- Modélisation des données
# * 4- Évaluation du modèle


# ### Modèle 1
# Appliquer GBC a commentaire en utilisant le reg et stop words pour faire tokenizer
# * 1- Préparation des données
# * 2- Construction du modèle
# * 3- Modélisation des données
# * 4- Evaluation du modèle

# importer les données nettoyées 
df = pd.read_csv('../data/reviews_trust_fr_VF.csv',index_col = 0)
# entrée : Commentaire
# cible : star (0, 1)
new_star = {1:0,2:0,3:1,4:1,5:1}
df.star.replace(new_star,inplace = True)
features = df.Commentaire
target = df.star.astype('int')
X_train_org,X_test_org,y_train, y_test = train_test_split(features, target, test_size = 0.2,random_state=20)

X_train_GBC_2 = X_train_org
X_test_GBC_2 = X_test_org

# traitement des mots vides #
# chargement fichier excel de stop words french en dataframe
df_stop_word_xls = pd.read_excel('../data/liste_no-stop-words_tokens_unique.xlsx', header=None)

# création de stop words set

# # liste selon le retour d'expérience projet
update_list_fr = list(df_stop_word_xls[0])

# initialisation de la variable des mots vides
stop_words = set()
stop_words.update(update_list_fr)

# Initialiser un objet vectorizer, garde aussi un mot avec plus 2 lettres
vectorizer = CountVectorizer(stop_words = stop_words,token_pattern = r"[a-zéèêàâîïàùçôëûæœ]{2,}" )

# Mettre à jour la valeur de X_train et X_test
X_train_GBC_2 = vectorizer.fit_transform(X_train_GBC_2).todense()
X_test_GBC_2 = vectorizer.transform(X_test_GBC_2).todense()

# print(vectorizer.vocabulary_)
GBC_2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train_GBC_2, y_train)

# sauvegarder le modèle pré-entrainé
#pickle.dump(GBC_2,open('../data/GBC_2.pickle','wb')) 
# sauvegarder le Vectoriser
#pickle.dump(vectorizer, open('../data/vectoriser_GBC_2','wb'))


# Calculer les prédictions 
y_pred_GBC_2 = GBC_2.predict(X_test_GBC_2)

# Calcul et affichage de classification_report
print(classification_report(y_test, y_pred_GBC_2) )

# Calcul et affichage de la matrice de confusion
confusion_matrix = pd.crosstab(y_test, y_pred_GBC_2, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(confusion_matrix)

"""

# 
# <hr style="border-width:2px;border-color:#75DFC1">
# <center><h1>3ème Modélisation avec pré-entrainement du corpus documents</h1></center>
# <hr style="border-width:2px;border-color:#75DFC1">
# 
# #### 1- passage du corpus sur un modèle pré-entrainé en langue Française provenant de wikipedia de 2.500.733 de mots vectorisés
# #### 2- modélisation et prédiction par SVM() à noyau RBF
# #### 3- modélisation et prédiction par GBC 
# #### 4- modélisation et prédiction par BernoulliNB()
# #### 5- modélisation et prédiction par les réseaux neurones (CNN, RNN..)
# #### 6- tableau de comprehension des erreurs de prédiction avec leur corpus original et filtré
# #### 7- traitement et prédiction d'un commentaire 'à la volée'
# #### 8- analyse finale et pistes d'améliorations


# importer les données nettoyées
data = pd.read_csv('../data/review_trust_fr_lemmantiser_word+2_VF.csv', sep=',')  
# à corriger +2
# test reviews_trust_fr_VF.csv 
data.drop(data.iloc[:,:1], axis=1, inplace=True)
data.sample(1)


# ## 2- passage du corpus sur un modèle pré-entrainé en langue Française provenant de wikipedia de 2.500.733 de mots vectorisés
# ### Chargement du modèle pré-entrainé de wikipedia2vec
# - TUTO d'aide pour charger une réprésentation pré-entraînée dans une langue souhaitée de wikipedia2vec
# 
#    - 1 aller sur le lien @  https://wikipedia2vec.github.io/wikipedia2vec/pretrained/#french
#    - 2 copier le lien d'adresse d'un des 4 fichiers souhaité en langue Française de la ligne
#      frwiki_20180420 (window=5, iteration=10, negative=15): 100d (bin) 100d (txt) 300d (bin) 300d (txt)
#    - 3 placer le lien sur notre IDE spyder ou jupyter
#    - 4 avec la souris sur ce chemin, faire CRL + clic => le fichier se télécharge = 1h pour le + petit de 450Mo nommé 100d(txt)
#    - 5 placer ce fichier dans un dossier et le décompresser => 1.9Go
#    - 6 placer ce fichier décompressé dans le dossier C:\\Users\\UserPC\\anaconda3\\envs\\spyder-env\\lib\\site-packages\\gensim\\test\\test_data\\
#      où spyder-env est mon environnement spécial spyder (pas obligatoire)
#    - 7 enfin lancer la cellule ci-dessous pour charger le fichier sur notre IDE et créer un vecteur
# 
# 

# Le chargement d'un objet gensim prend 6 à 8 mins chaque fois, le modèle **trained** est sauvegardé et utilisé dans la suite
# 

# si c'est la première fois et que vous n'avez pas récupéré le trained.pickle, lancez les codes. sinon charger directement le modèle 
'''
# chargement d'un objet gensim de type models.keyedvectors de taille 2.500.733
start = time()

trained = load_word2vec_format('frwiki_20180420_100d.txt.bz2', binary=False)    # 6 à 8 min de chargement, patience...

end = time()
print("temps chargement modele pré-entrainé = ", end-start)
# à voir absolument le parametre unicode_errors='ignore' ou 'replace' qui sera utile par la suite pour éviter des ereurs en chargement

print(trained.vectors.shape) 
'''
'''
# sauvegarder le modèle pré-entrainé
pickle.dump(trained,open('trained.pickle','wb')) 
'''


# charger le modèle pré-entrainé sauvegardé
trained = pickle.load(open('../data/Modeles/trained.pickle','rb'))


# Test du model pré-entrainé
# tests des tokens du model pré-entrainé pour voir comme ça marche

# test de similarité du token 'papa'
print(trained.most_similar(['papa']))  # 'mamam' à 78.67% puis 'papone à 70.87% etc.....

# exemple commun de test calculatoire sur les token : +A - C => +B - ?  avec A=roi, C = homme, B=femme => ? = reine solution
print(trained.most_similar(positive=['roi', 'femme'], negative=['homme']))  # => 'reine' 1er choix à 86.49% puis 'régente' etc...


# ## Fonction my_doc2vec() de pré-traitement

# fonction pour transformer un document ( ici une ligne de commentaire) en vecteur à partir des tokens qui le compose
# entrée : line_com lignes commentaires à traiter
#          model pré-entrainé
# sortie : vecteur représentant le document

def my_doc2vec(doc, trained):
    # dimension de représentation
    p = trained.vectors.shape[1]   # p = 100
    # intit du vecteur
    vec = np.zeros(p)  # array (100,)
    # nbre de tokens trouvés
    nb=0
    # traitement de chaque token de la ligne de commentaire
    for tk in doc:
        # ne traiter que les tokens reconnus
        try:
            values = trained[tk] # on récupère le vecteur du token concerné
            vec = vec + values   # on incrémente ce vecteur dans vec
            nb = nb + 1.0        # le compteur de token d'incrémente
        except:
            pass  # pour lever l'erreur si aucun token trouvé dans notre modele pré-entrainé
        # moyenne des valeurs uniquement si on a des tokens reconnus
    if (nb > 0.0):
        vec = vec/nb
    return vec  # renvoie le vecteur moyenné ou un vecteur nul si aucun token trouvé


# ## Pré-traitement du corpus

# préparer la liste corpus_sw pour modèle 
corpus_sw = data['no_stop_words'].tolist()
# corpus_sw = data['words+3'].tolist() 
# 
# traiter les documents du corpus

docsVec = list()
# pour chaque ligne du corpus nettoyé    
for doc in corpus_sw:
    # calcul de son vecteur
    vec = my_doc2vec(doc,trained)
    # ajouter dans la liste
    docsVec.append(vec)    
# transfromer en array numpy
matVec = np.array(docsVec)
print(matVec.shape)


# transformer en dataframe
df = pd.DataFrame(matVec, columns=["v"+str(i+1) for i in range(matVec.shape[1])])
# ajouter la classe target 'star'
df['label'] = data.star
df.head(3)


# ## partition en train_set et test_set

# partition en train_set et test_set
dfTrain, dfTest = train_test_split(df, test_size=0.2, stratify=df.label, random_state=20)
# que 2 instances au lieu de 4 car taget est dans df !

"""
# # 2- Modèle SVM 
# 
# * -Modélisation par SVM, prédiction et évaluation en test


# SVM avec noyau RBF par défaut
SVM = SVC(C=100, gamma=1,random_state=0)  
# ces paramètres ont été trouvés par GridSearchCV sur quelques valeurs seulement
# nous donnons le code pour cela ci-dessous mais à lancer avec parcimonie puisque les temps de calculs ont sont multipliés ! 
SVM.fit(dfTrain[dfTrain.columns[:-1]], dfTrain.label)

# sauvegarder le modèle entrainé
#pickle.dump(SVM,open('SVM.pickle','wb')) 


# prediction en test
y_pred_SVM = SVM.predict(dfTest[dfTest.columns[:-1]])

# évaluation des performances
print(classification_report(dfTest.label, y_pred_SVM))

# Calcul et affichage de la matrice de confusion
confusion_matrix = pd.crosstab(dfTest.label, y_pred_SVM, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(confusion_matrix)


# recherche des best parametres avec gridsearshCV
#from sklearn.model_selection import GridSearchCV

#param_grid = {'C':[50, 100], 'kernel': ['rbf'], 'gamma': [1, 10]}  # recherche des bons hyperparams
#grid = GridSearchCV(SVC(), param_grid) # création de la grille de recherche sur le model SVC()

#grille = grid.fit(X_train, y_train)
# affichage des résultats test de GridSearchCV
#print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']]) 
#print("grid.best_params :", grid.best_params_)
#svm_best= grid.best_estimator_

"""
# 3- MODELE ANN ARTIFICEL NEURONAL NETWORK

# rependre les données 
# transformer en dataframe
X = pd.DataFrame(matVec, columns=["v"+str(i+1) for i in range(matVec.shape[1])])
# ajouter la classe target 'star'
y = data.star

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20) 

"""
inputs = Input(shape = (100), name = "Input") # couche d'entrée qui contient les dimensions de nos données X en entrée, ici 100 car 100 colonnes
dense1 = Dense(units = 50, activation = "tanh", name = "Dense_1")  # pas besoin des dimensions shape pour les couches suivantes,
dense2 = Dense(units = 20, activation = "tanh", name = "Dense_2")
dense3 = Dense(units = 8, activation = "tanh", name = "Dense_3")
dense4 = Dense(units = 2, activation = "softmax", name = "Dense_4")
"""

inputs = Input(shape = (100))
dense_1 = Dense(units = 64, activation = "relu")
dropout_2 = Dropout(rate=0.1)
dense_3 = Dense(units =32, activation = "relu")
dropout_4 = Dropout(rate=0.1)
dense_5 = Dense(units = 2, activation = "sigmoid")

x=dense_1(inputs)  # -> x de type engine.keras_tensor.KerasTensor
x=dropout_2(x)
x=dense_3(x)
x=dropout_4(x)
outputs=dense_5(x)  # tous les x suivant et le outputs sont du meme type que le 1er x
# - Les commandes suivantes permettent de finaliser la définition du modèle et d'en afficher la structure.
ANN = Model(inputs = inputs, outputs = outputs,name = "ANN")
ANN.summary()


# compilation du modele avec "binary_crossentropy" ne marche pas !! 
ANN.compile(loss = "sparse_categorical_crossentropy",
            optimizer = tf.keras.optimizers.Adam(),
            metrics = ["accuracy"])

# AUTOMATIC REDUCTION OF THE LEARNING RATE
lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',          # Metric to control
                                         patience = 5,                  # Number of epochs to wait before decreasing the learning rate
                                         factor = 0.1,                  # Percentage learning rate is decreased
                                         verbose = 2,                   # Number of informations displayed during training
                                         mode = 'min')                  # Indicate the metric decrease

# EARLY STOPPING
early_stopping = callbacks.EarlyStopping(monitor = 'val_loss',          # Metric to control
                                         patience = 13,                 # Number of epochs to wait before stopping the training
                                         mode = 'min',                  # Indicate the metric decrease
                                         restore_best_weights = True)   # To restore the weights of the best epoch

BATCH_SIZE = 128

# Initialization of a timer
t0 = time()
ANN.fit(X_train, y_train,
        validation_data=(X_test,y_test),
        epochs=100,
        steps_per_epoch = (len(X_train)) // (BATCH_SIZE * 2),          # Number of steps per Epoch
        validation_steps = (len(X_test)) // (BATCH_SIZE * 2),          # Number of iterations during the test  
        callbacks = [lr_plateau, early_stopping],                      # Callbacks
        workers = -1)                                                  # Using all processors

# Calulation of training time
t1 = time() - t0
print("Training time in {} secondes".format(t1))

# sauvegarder le modèle entrainé
#pickle.dump(ANN,open('../data/ANN.pickle','wb'))


test_pred = ANN.predict(X_test)  # => dim (5160, 2) où n =dim[0]de X_test et 2 = dim units dense_last
y_pred_class = np.argmax(test_pred,axis=1) # on prend l'index du max de chaque ligne prédite
#y_pred_class = np.where(test_pred[:,1] > 0.52, 1, 0)

y_test_class = y_test  # autre nom a y_test pour la suite classification_report et confusion_matrix


print(classification_report(y_test_class,y_pred_class)) 
print(confusion_matrix(y_test_class,y_pred_class))


# # Prédiction 
# modèle à choisir
"""
GBC_3
SVM
ANN
"""
text = "produit cassé, livraison lente, pas de réponse de service client"

"""
# # Fonction : text préparation et prédiction 

# !python -m spacy download fr_core_news_md
nlp = spacy.load('fr_core_news_md')
def prediction(text, model):
    my_doc = text.lower()
    my_doc_tk = word_tokenize(my_doc)
    def lemms(corpus_tk):    
        doc = nlp(" ".join(corpus_tk))
        lemm = [token.lemma_ for token in doc]
        return lemm
    my_doc_lem = lemms(my_doc_tk)
    def stop_words_filtering(mots, stop_words) : 
        tokens = []
        for mot in mots:
            if mot not in stop_words: 
                tokens.append(mot)
        return tokens
    df_stop_word_xls = pd.read_excel('../data/liste_no-stop-words_tokens_unique.xlsx', header=None)
    update_list_fr = list(df_stop_word_xls[0])
    # initialisation de la variable des mots vides
    stop_words = set()
    stop_words.update(update_list_fr)
    my_doc_sw = stop_words_filtering(my_doc_lem,stop_words)
    my_vec = my_doc2vec(my_doc_sw,trained)
    
#     print(my_doc_sw)
    pred_my_doc = model.predict(my_vec.reshape(1,-1))
    return pred_my_doc


print(prediction(text,SVM))
print(prediction(text,GBC_3))
print(prediction(text,ANN))
# 1 - positive
# 0 - negative 
# ANN nous donne un résultat proba 


"""
"""
# # Modèle 7 - Réseau Neurone

# chargement du bon csv des données en français 
df = pd.read_csv('../data/review_trust_fr_lemmantiser_word+2_VF.csv', sep=',') 
# utiliser une vectorization proposée par keras
df.drop(df.iloc[:,:1], axis=1, inplace=True)
df.head(2)


# # enlever les informations inutiles
# df = df.drop(['source', 'company','date','tokens','words+2','lemmatiser'], axis = 1)


# df.head(2)


df['comment'] = df['no_stop_words'].apply(lambda x : x.replace(',',' ').replace("'","").replace("  "," ").replace("[","").replace("]",""))
df['comment'][0]


# liste de paramètres
# DATASET
TRAIN_SIZE = 0.8
# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024


df = df[['comment','star']]
df =df.rename({'comment':'text', 'star':'target'},axis = 1)
df.head()


# split train and test
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state = 42)
print('train size: ', len(df_train))
print('test size: ', len(df_test))


# ## Tokenize text

# package utilisé
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
# différente version keras le nom de keras_precessing.sequence peut être différent comme keras.precessing.sequence
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


get_ipython().run_cell_magic('time', '', 'tokenizer = Tokenizer()\ntokenizer.fit_on_texts(df_train.text)\nvocab_size = len(tokenizer.word_index)+1\nprint("total words ", vocab_size)\n')


get_ipython().run_cell_magic('time', '', 'x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)\nx_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)\n# pad_sequences: Pads sequences to the same length\n')


y_train = df_train.target.values.reshape(-1,1)
y_test = df_test.target.values.reshape(-1,1)
print('y_train shape ',y_train.shape)
print('y_test shape ', y_test.shape)


# vérification des dimentions
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)


# ## Embedding layer

# ### importer le résultat - word2vec 

from gensim.models.keyedvectors import load_word2vec_format
from gensim.test.utils import datapath
w2v_model =  pickle.load(open('trained.pickle','rb'))


embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.key_to_index:
        embedding_matrix[i] = w2v_model.key_to_index[word]
print(embedding_matrix.shape)
        


embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)


# ## build model

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# ## compile model

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


# ## Callbacks

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]


# # Train

get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train,\n                    batch_size=BATCH_SIZE,\n                    epochs=EPOCHS,\n                    \n                    validation_split=0.1,\n                    verbose=1,\n                    callbacks=callbacks)\n')


get_ipython().run_cell_magic('time', '', 'score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)\nprint()\nprint("ACCURACY:",score[1])\nprint("LOSS:",score[0])\ny_pred= model.predict(x_test)\n')

"""