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
#get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
from time import time
import time 
import pickle
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
#import langid                      # langid non plus
#from wordcloud import WordCloud
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
import tensorflow
from tensorflow.keras.layers import Input, Dense #Pour instancier une couche Dense et une d'Input
from tensorflow.keras.models import Model
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
print('Tensorflow            : ' + tensorflow.__version__)
print('Spacy                 : ' + spacy.__version__)


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


# # 4- Modélisation des données
# 
# Il y a 3 modèles d'apprentissage appliqué dans le projet. Les données d'entrées sont différentes. 
# Notre rapport de projet en a sélectionné seulement 7 pour éviter les redondances inutiles
# 
# Pour faire la modélisation, les étapes principales : 
# * 1- Préparation des données
# * 2- Construction du modèle
# * 3- Modélisation des données
# * 4- Évaluation du modèle

# # Modèle 1 - Logistic Regression
# 
 

# importer les données nettoyées
df = pd.read_csv('../data/reviews_trust_fr_VF.csv',index_col = 0)
df.head()


# ##  1- Préparation des données
# 
# Les étapes principales : 
# 
# * a- raffiner les commentaires 
# * b- traiter les dates
# * c- traiter les star, company, source
# 
# ### a- raffiner les commentaires

def find_exclamation(Commentaire):
    r = re.compile(r"\!")
    exclamation = r.findall(Commentaire)
    return len(exclamation)

def find_interogation(Commentaire):
    r = re.compile(r"\?")
    interogation = r.findall(Commentaire)
    return len(interogation)

def find_etc(Commentaire):
    r = re.compile(r"\.{2,}")
    etc = r.findall(Commentaire)
    return len(etc)

df['exclamation'] = df.Commentaire.apply(lambda x: find_exclamation(x))
df['interogation'] = df.Commentaire.apply(lambda x: find_interogation(x))
df['etc'] = df.Commentaire.apply(lambda x: find_etc(x))
df['nb_caracter'] = df.Commentaire.apply(len)


# ### b- traiter les dates

df['date'] = pd.to_datetime(df['date'],utc = True)
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['weekday'] = pd.to_datetime(df['date']).dt.weekday
df.drop('date', axis = 1, inplace = True)


# ### c- traiter les star, company, source
# 
# * positive = 1 corresepond star 3, 4 et 5
# * négative = 0 correspond start 1 et 2


new_star = {1:0,2:0,3:1,4:1,5:1}
df.star.replace(new_star,inplace = True)


# observation sur les entrées
# objectifs : vérifier si les entrées ont un sens 
# display(df[['star', 'exclamation', 'chain_exclamation', 'interogation', 'nb_caracter']].groupby(['star']).sum())
display(df[['star', 'exclamation', 'interogation', 'nb_caracter']].groupby(['star']).sum())


df= df.join(pd.get_dummies(df['source'],prefix = 'source'))
df= df.join(pd.get_dummies(df['company'],prefix = 'company'))
df.drop(['source','company'], axis= 1, inplace = True)


df.head()
df.to_csv('reviews_trust_fr_LR.csv')
# sauvegarder les données pour alimenter le 3ème modèle BernoulliNB()


# Jusqu'à ici, la quantification des informations (*data, commentaire, source, star*) est finie. 
# 
# ### 2- Construction du modèle
# ### 3- Modélisation des données
# - entrées (data/X): 'exclamation', 'chain_exclamation', 'interogation', 'nb_caracter','source','company','year','month','weekday'
#    
# - sortie (target/Y): 'star'

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Commentaire','star','index_org','star_org'], axis=1), df.star, test_size=0.2, random_state=20)


logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)


# ### 4- Evaluation du modèle

# les score : 
print('Score :',logreg.score(X_test, y_test))


# cross table
y_pred = logreg.predict(X_test)
pd.crosstab(y_test, y_pred, rownames = ['real'], colnames =['predict'])


print(classification_report(y_test, y_pred))


# ### Création et sauvegarde du tableau des erreurs de prédiction pour analyse ultérieure

# on récupère les données complètes du test_set avec indexs et star d'origine
# création d'un dataframe vide
df_analyse = pd.DataFrame()  
# recupérer tou le y_test en renommant la colonne star par star_1 pour modele 1
df_analyse['star_1'] = y_test     
# ajouter une colonne d'index ordonnée de 0 à taille du df et déplacer les index d'erreurs dans une colonne du df
df_analyse.reset_index(inplace=True, drop=False)
# renommer la colonne d'index d'erreurs par index_1 pour modele 1
df_analyse.rename(columns={'index':'index_1'}, inplace=True)
# ajouter au df la colonnede prédiction réalisée sur y_test
df_analyse['pred_1'] = pd.Series(y_pred)                                        


# filtrage des erreurs de prediction par boolean indexing
 # création d'une liste boolean vérifiant sur les prédiction sont bonnes ou mauvaises       
bool_pred = [True if (df_analyse.iloc[i][1]==df_analyse.iloc[i][2]) else False for i in range(df_analyse.shape[0])]
# ajout de la liste du résultat booleen dans le tableau d'analyse des erreurs
df_analyse['bool_1'] = pd.DataFrame(bool_pred)
# création d'un dataframe des seules erreurs du modele
df_verification = df_analyse.loc[df_analyse['bool_1']==False]   # Parfait 386 erreurs aussi ici de pred facilement trouvable avec vrais index
# effacement des index non originaux copiés et sans intéret
df_verification.reset_index(inplace=True, drop=True)  


# Rajouter la colonne de commentaires originaux + colones commentaires filtrés finaux à df_ verification
list_com_vo = []
for i in range(df_verification.shape[0]):
    find_index = df_verification['index_1'][i]
    list_com_vo.append(df['Commentaire'][find_index])
    
# ajout commentaires originaux    
df_verification['comment_vo'] = pd.Series(list_com_vo)       # Parfait , les commentaires sont aux bons index !


# affichage du tableau comparatif des erreurs réalisé
print("df_verification.shape = ", df_verification.shape)
df_verification.head(10)


# ### Nous retrouvons bien les 386 erreurs identifiées dans la matrice de confusion du modele 1

# # Modèle 2- Gradient Boosting Classifier
# * Modèle 2.1- sans utiliser le regex et stop words pour faire tokenizer
# * Modèle 2.2- en utilisant le regex et stop words pour faire tokenizer
# 

# importer les données nettoyées 
df = pd.read_csv('reviews_trust_fr_VF.csv',index_col = 0)
# entrée : Commentaire
# cible : star (0, 1)
new_star = {1:0,2:0,3:1,4:1,5:1}
df.star.replace(new_star,inplace = True)
features = df.Commentaire
target = df.star.astype('int')
X_train_org,X_test_org,y_train, y_test = train_test_split(features, target, test_size = 0.2,random_state=20)


# **NOTE**: Ici deux modèles GBC avec ou sans filtrage de commentaires sont utilisés. Pour comparer les résultats, les données d'entrées sont les même.

# ### Modèle 2.1 
# Appliquer GBC a *commentaire* sans utiliser le regex et stop words pour faire tokenizer
# * 1- Préparation des données
# * 2- Construction du modèle
# * 3- Modélisation des données
# * 4- Evaluation du modèle

get_ipython().run_cell_magic('time', '', "\nvectorizer = CountVectorizer()\n\nX_train_GBC_1 = X_train_org # utiliser les commentaires origines pour modéliser\nX_test_GBC_1 = X_test_org   # utiliser les commentaires origines pour modéliser\n\n# Mettre à jour la valeur de X_train et X_test\nX_train_GBC_1 = vectorizer.fit_transform(X_train_GBC_1).todense()\nX_test_GBC_1 = vectorizer.transform(X_test_GBC_1).todense()\n\nGBC_1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train_GBC_1, y_train)\n\n# sauvegarder le modèle pré-entrainé\npickle.dump(GBC_1,open('GBC_1.pickle','wb')) \n")


# sauvegarder le Vectoriser 
pickle.dump(vectorizer, open('vectoriser_GBC_1','wb'))


get_ipython().run_cell_magic('time', '', "# Calculer les prédictions \ny_pred_GBC_1 = GBC_1.predict(X_test_GBC_1)\n\n#  Importer la classe classification_report\nfrom sklearn.metrics import classification_report\n\n# Calcul et affichage de classification_report\nprint(classification_report(y_test, y_pred))\n\n# Calcul et affichage de la matrice de confusion\nconfusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])\nconfusion_matrix\n")


# ### Création et sauvegarde du tableau des erreurs de prédiction pour analyse ultérieure

# on récupère les données complètes du test_set avec indexs et star d'origine
# création d'un dataframe vide
df_analyse = pd.DataFrame()  
# recupérer tou le y_test en renommant la colonne star par star_2 pour modele 2
df_analyse['star_2'] = y_test     
# ajouter une colonne d'index ordonnée de 0 à taille du df et déplacer les index d'erreurs dans une colonne du df
df_analyse.reset_index(inplace=True, drop=False)
# renommer la colonne d'index d'erreurs par index_2 pour modele 2
df_analyse.rename(columns={'index':'index_2'}, inplace=True)
# ajouter au df la colonnede prédiction réalisée sur y_test
df_analyse['pred_2'] = pd.Series(y_pred)                                        


# filtrage des erreurs de prediction par boolean indexing
 # création d'une liste boolean vérifiant sur les prédiction sont bonnes ou mauvaises       
bool_pred = [True if (df_analyse.iloc[i][1]==df_analyse.iloc[i][2]) else False for i in range(df_analyse.shape[0])]
# ajout de la liste du résultat booleen dans le tableau d'analyse des erreurs
df_analyse['bool_2'] = pd.DataFrame(bool_pred)
# création d'un dataframe des seules erreurs du modele
df_verification = df_analyse.loc[df_analyse['bool_2']==False]   # Parfait 386 erreurs aussi ici de pred facilement trouvable avec vrais index
# effacement des index non originaux copiés et sans intéret
df_verification.reset_index(inplace=True, drop=True)  


# Rajouter la colonne de commentaires originaux + colones commentaires filtrés finaux à df_ verification
list_com_vo = []
for i in range(df_verification.shape[0]):
    find_index = df_verification['index_2'][i]
    list_com_vo.append(df['Commentaire'][find_index])
    
# ajout commentaires originaux    
df_verification['comment_vo'] = pd.Series(list_com_vo) 

# affichage du tableau comparatif des erreurs réalisé
print("df_verification.shape = ", df_verification.shape)
df_verification.head(10)


# sauvegarde des erreurs à analyser sur le model2
df_verification.to_csv('sav_verif_erreurs_model2.csv')


# ### Prédiction à la demande sur GBD 1 (en direct) 

# prediction à la demande
# charger le modèle pré-entrainé sauvegardé
GBC_1 = pickle.load(open('GBC_1.pickle','rb'))
# text = "la livraison est lente, le produit est abîmé"
text = "mauvais produit"
text= pd.Series(text)
text_vec = vectorizer.transform(text).todense()
print(text_vec)
pred = GBC_1.predict(text_vec)
str(pred)


# - mauvaise prédiction => erreur dans le code ? ou normal car peu de mots => à creuser avec les tableaux d'erreurs et faire d'autres tests

# ### Modèle 2.2 
# Appliquer GBC a commentaire en utilisant le reg et stop words pour faire tokenizer
# * 1- Préparation des données
# * 2- Construction du modèle
# * 3- Modélisation des données
# * 4- Evaluation du modèle

get_ipython().run_cell_magic('time', '', 'X_train_GBC_2 = X_train_org\nX_test_GBC_2 = X_test_org\n\n# traitement des mots vides #\n# chargement fichier excel de stop words french en dataframe\ndf_stop_word_xls = pd.read_excel(\'liste_no-stop-words_tokens_unique.xlsx\', header=None)\n\n# création de stop words set\n\n# # liste selon le retour d\'expérience projet\nupdate_list_fr = list(df_stop_word_xls[0])\n\n# initialisation de la variable des mots vides\nstop_words = set()\nstop_words.update(update_list_fr)\n\n# Initialiser un objet vectorizer, garde aussi un mot avec plus 2 lettres\nvectorizer = CountVectorizer(stop_words = stop_words,token_pattern = r"[a-zéèêàâîïàùçôëûæœ]{2,}" )\n\n# Mettre à jour la valeur de X_train et X_test\nX_train_GBC_2 = vectorizer.fit_transform(X_train_GBC_2).todense()\nX_test_GBC_2 = vectorizer.transform(X_test_GBC_2).todense()\n\n# print(vectorizer.vocabulary_)\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.ensemble import GradientBoostingClassifier\nGBC_2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train_GBC_2, y_train)\n\n# sauvegarder le modèle pré-entrainé\npickle.dump(GBC_2,open(\'GBC_2.pickle\',\'wb\')) \n# sauvegarder le Vectoriser\npickle.dump(vectorizer, open(\'vectoriser_GBC_2\',\'wb\'))\n')


get_ipython().run_cell_magic('time', '', "# Calculer les prédictions \ny_pred_GBC_2 = GBC_2.predict(X_test_GBC_2)\n#  Importer la classe classification_report\nfrom sklearn.metrics import classification_report\n\n# Calcul et affichage de classification_report\nprint(classification_report(y_test, y_pred) )\n\n# Calcul et affichage de la matrice de confusion\nconfusion_matrix = pd.crosstab(y_test, y_pred_GBC_2, rownames=['Classe réelle'], colnames=['Classe prédite'])\nconfusion_matrix\n")


# # Modèle 3 - Naive Bayes classifier for multivariate Bernoulli models
#  ### -- BernoulliNB()

# importer les données nettoyées 
df = pd.read_csv('reviews_trust_fr_LR.csv',index_col = 0)
# utiliser les données quantitavie du Logistic regression


df.head()


# bag of words 
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import stopwords
# from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB,ComplementNB,CategoricalNB

# Split data in train and validate dataset.
df_train, df_test, y_train, y_test = train_test_split(df.drop(['star','index_org','star_org'], axis=1), 
                                                      df.star, test_size=0.2, random_state=20)


# traitement des mots vides #
# chargement fichier excel de stop words french en dataframe
df_stop_word_xls = pd.read_excel('liste_no-stop-words_tokens_unique.xlsx', header=None)

# création de stop words set

# # liste selon le retour d'expérience projet
update_list_fr = list(df_stop_word_xls[0])

# initialisation de la variable des mots vides
stop_words = set()
stop_words.update(update_list_fr)

# Bag of words, garde le mot avec au moins deux lettres
vectorizer = CountVectorizer(stop_words=stop_words, token_pattern = r"[a-zéèêàâîïàùçôëûæœ]{2,}",max_features=5000)

X_train_text = vectorizer.fit_transform(df_train.Commentaire)
X_test_text = vectorizer.transform(df_test.Commentaire)

from scipy.sparse import hstack
# hstack(): concater le text vectoriser (x_train_text) avec les autres informations quantitative 
# (comme data, year, nombre d'exclamation ) 
X_train = hstack((X_train_text, df_train.drop('Commentaire', axis=1).values))
X_test = hstack((X_test_text, df_test.drop('Commentaire', axis=1).values))

from sklearn.naive_bayes import BernoulliNB
ber = BernoulliNB()
ber.fit(X_train, y_train)

# sauvegarder le modèle entrainé
pickle.dump(ber,open('ber.pickle','wb')) 
# sauvegarder le Vectoriser
pickle.dump(vectorizer, open('vectoriser_ber','wb'))

# @fred: je ne pense pas que nous allons utiliser ce modèle pour predire une commentaire, 
# car il faut reproduire les données statistiques comme data, day etc
# à supprimer après ta lecture merci :) 

print('Score :',ber.score(X_test, y_test))
# Calculer les prédictions 
y_pred_ber = ber.predict(X_test)
# Calcul et affichage de la matrice de confusion
confusion_matrix = pd.crosstab(y_test, y_pred_ber, rownames=['Classe réelle'], colnames=['Classe prédite'])
confusion_matrix


# Calcul et affichage de classification_report
print(classification_report(y_test, y_pred) )


# ### Création et sauvegarde du tableau des erreurs de prédiction pour analyse ultérieure

# on récupère les données complètes du test_set avec indexs et star d'origine
# création d'un dataframe vide
df_analyse = pd.DataFrame()  
# recupérer tou le y_test en renommant la colonne star par star_3 pour modele 3
df_analyse['star_3'] = y_test     
# ajouter une colonne d'index ordonnée de 0 à taille du df et déplacer les index d'erreurs dans une colonne du df
df_analyse.reset_index(inplace=True, drop=False)
# renommer la colonne d'index d'erreurs par index_2 pour modele 2
df_analyse.rename(columns={'index':'index_3'}, inplace=True)
# ajouter au df la colonnede prédiction réalisée sur y_test
df_analyse['pred_3'] = pd.Series(y_pred)                                        


# filtrage des erreurs de prediction par boolean indexing
 # création d'une liste boolean vérifiant sur les prédiction sont bonnes ou mauvaises       
bool_pred = [True if (df_analyse.iloc[i][1]==df_analyse.iloc[i][2]) else False for i in range(df_analyse.shape[0])]
# ajout de la liste du résultat booleen dans le tableau d'analyse des erreurs
df_analyse['bool_3'] = pd.DataFrame(bool_pred)
# création d'un dataframe des seules erreurs du modele
df_verification = df_analyse.loc[df_analyse['bool_3']==False]   # Parfait 386 erreurs aussi ici de pred facilement trouvable avec vrais index
# effacement des index non originaux copiés et sans intéret
df_verification.reset_index(inplace=True, drop=True)  


# Rajouter la colonne de commentaires originaux + colones commentaires filtrés finaux à df_ verification
list_com_vo = []
for i in range(df_verification.shape[0]):
    find_index = df_verification['index_3'][i]
    list_com_vo.append(df['Commentaire'][find_index])
    
# ajout commentaires originaux    
df_verification['comment_vo'] = pd.Series(list_com_vo)


# affichage du tableau comparatif des erreurs réalisé
print("df_verification.shape = ", df_verification.shape)
df_verification.head(10)


# sauvegarde des erreurs à analyser sur le model3
df_verification.to_csv('sav_verif_erreurs_model3.csv')


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
data = pd.read_csv('review_trust_fr_lemmantiser_word+2_VF.csv', sep=',')  
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
trained = pickle.load(open('trained.pickle','rb'))


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


# # 4- Modèle SVM 
# 
# * -Modélisation par SVM, prédiction et évaluation en test


# SVM avec noyau RBF par défaut
SVM = SVC(C=100, gamma=1,random_state=0)  
# ces paramètres ont été trouvés par GridSearchCV sur quelques valeurs seulement
# nous donnons le code pour cela ci-dessous mais à lancer avec parcimonie puisque les temps de calculs ont sont multipliés ! 
SVM.fit(dfTrain[dfTrain.columns[:-1]], dfTrain.label)

# sauvegarder le modèle entrainé
pickle.dump(SVM,open('SVM.pickle','wb')) 


# prediction en test
y_pred_SVM = SVM.predict(dfTest[dfTest.columns[:-1]])

# évaluation des performances
print(classification_report(dfTest.label, y_pred_SVM))

# Calcul et affichage de la matrice de confusion
confusion_matrix = pd.crosstab(dfTest.label, y_pred_SVM, rownames=['Classe réelle'], colnames=['Classe prédite'])
confusion_matrix


# recherche des best parametres avec gridsearshCV
#from sklearn.model_selection import GridSearchCV

#param_grid = {'C':[50, 100], 'kernel': ['rbf'], 'gamma': [1, 10]}  # recherche des bons hyperparams
#grid = GridSearchCV(SVC(), param_grid) # création de la grille de recherche sur le model SVC()

#grille = grid.fit(X_train, y_train)
# affichage des résultats test de GridSearchCV
#print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']]) 
#print("grid.best_params :", grid.best_params_)
#svm_best= grid.best_estimator_


# ### Création et sauvegarde du tableau des erreurs de prédiction pour analyse ultérieure

# on récupère les données complètes du test_set avec indexs et star d'origine
# création d'un dataframe vide
df_analyse = pd.DataFrame()  
# recupérer tout le y_test en renommant la colonne star par star_4 pour modele 4
df_analyse['star_4'] = dfTest.label     
# ajouter une colonne d'index ordonnée de 0 à taille du df et déplacer les index d'erreurs dans une colonne du df
df_analyse.reset_index(inplace=True, drop=False)
# renommer la colonne d'index d'erreurs par index_2 pour modele 2
df_analyse.rename(columns={'index':'index_4'}, inplace=True)
# ajouter au df la colonnede prédiction réalisée sur y_test
df_analyse['pred_4'] = pd.Series(y_pred_SVM)                                        


# filtrage des erreurs de prediction par boolean indexing
 # création d'une liste boolean vérifiant sur les prédiction sont bonnes ou mauvaises       
bool_pred = [True if (df_analyse.iloc[i][1]==df_analyse.iloc[i][2]) else False for i in range(df_analyse.shape[0])]
# ajout de la liste du résultat booleen dans le tableau d'analyse des erreurs
df_analyse['bool_4'] = pd.DataFrame(bool_pred)
# création d'un dataframe des seules erreurs du modele
df_verification = df_analyse.loc[df_analyse['bool_4']==False]   # Parfait 386 erreurs aussi ici de pred facilement trouvable avec vrais index
# effacement des index non originaux copiés et sans intéret
df_verification.reset_index(inplace=True, drop=True)  


# Rajouter la colonne de commentaires originaux + colones commentaires filtrés finaux à df_ verification
list_com_vo = []
for i in range(df_verification.shape[0]):
    find_index = df_verification['index_4'][i]
    list_com_vo.append(data['Commentaire'][find_index])
    
# ajout commentaires originaux    
df_verification['comment_vo'] = pd.Series(list_com_vo)       # Parfait , les commentaires sont aux bons index !

"""
""" pas de stop words fait sur ce modele3 donc ci-dessous inutile                                   
list_com_filtre = []
for i in range(df_verification.shape[0]):
    find_index = df_verification['index_1'][i]
    list_com_filtre.append(df['no_stop_words'][find_index])
    
# ajout commentaires traités   
df_verification['no_stop_words'] = pd.Series(list_com_filtre) 
"""

"""
# affichage du tableau comparatif des erreurs réalisé
print("df_verification.shape = ", df_verification.shape)
df_verification.head(10)


# sauvegarde des erreurs à analyser sur le model4
df_verification.to_csv('sav_verif_erreurs_model4.csv')  # 534 erreurs


# ### Tentative de prédiction sur un text à la demande

# prediction
# charger le modèle pré-entrainé sauvegardé
SVM_wiki = pickle.load(open('SVM.pickle','rb'))
# text = "la livraison est lente, le produit est abîmé"
text2 = "mauvais produit"

text_vec2 = my_doc2vec(text2, trained)  # text vectorisé de shape (100,)

# transformation du text vectorisé en shape (1, 100) avec les noms de colonnes v1 ...v 100 comme dans le modèle pré-entrainé
# récupérer la liste des noms de colonnes du dfTest entrainé
list_col = list(dfTest.columns[:-1])
# conversion du vecteur text en dataframe + transposé => shape (1, 100)
text_vector = pd.DataFrame(my_doc2vec(text2, trained)).T  # colonnes de 0 à 99 et ligne 0 du texte vectorisé !!
# renommer les noms des colonnes avec list_col
text_vector.columns = list_col

print(text_vec2)

# prédiction du text vectorisé
pred_wiki_direct_SVM = SVM_wiki.predict(text_vector)

print("La prédiction du text : ", text2, " est de ", pred_wiki_direct_SVM)



# ici aussi, mauvaise prédiction => faire d'autres test, voir avec lestableaux d'erreurs, vérifier que le code est correct


# # 5- Modèle GBC avec word2vec wikipedia

#  GBC appliquer sur les données vectoriées avec filtrage stop word

GBC_3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(dfTrain[dfTrain.columns[:-1]], dfTrain.label)

# Calculer les prédictions 
y_pred_GBC_3 = GBC_3.predict(dfTest[dfTest.columns[:-1]])

# sauvegarder le modèle entrainé
pickle.dump(GBC_3,open('GBC_3.pickle','wb'))


# Calcul et affichage de classification_report
print('Gradient Boosting Classifier avec données vectorizées: ')
print(classification_report(dfTest.label, y_pred_GBC_3))

# Calcul et affichage de la matrice de confusion
confusion_matrix = pd.crosstab(dfTest.label, y_pred_GBC_3, rownames=['Classe réelle'], colnames=['Classe prédite'])
confusion_matrix


# # Modèle 6 - Réseau Neurone 1
# ## *ANN ARTIFICEL NEURONAL NETWORK*
# 

# rependre les données 
# transformer en dataframe
X = pd.DataFrame(matVec, columns=["v"+str(i+1) for i in range(matVec.shape[1])])
# ajouter la classe target 'star'
y = data.star

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 


# ## Construire le modèle 

# MODELE ANN ARTIFICEL NEURONAL NETWORK --  KERAS -----------------------------------
inputs = Input(shape = (100), name = "Input") # couche d'entrée qui contient les dimensions de nos données X en entrée, ici 100 car 100 colonnes

dense1 = Dense(units = 50, activation = "tanh", name = "Dense_1")  # pas besoin des dimensions shape pour les couches suivantes,
dense2 = Dense(units = 20, activation = "tanh", name = "Dense_2")
dense3 = Dense(units = 8, activation = "tanh", name = "Dense_3")
dense4 = Dense(units = 2, activation = "softmax", name = "Dense_4")
x=dense1(inputs)  # -> x de type engine.keras_tensor.KerasTensor
x=dense2(x)
x=dense3(x)
outputs=dense4(x)  # tous les x suivant et le outputs sont du meme type que le 1er x
# - Les commandes suivantes permettent de finaliser la définition du modèle et d'en afficher la structure.
ANN = Model(inputs = inputs, outputs = outputs,name = "ANN")
ANN.summary()


# compilation du modele avec "binary_crossentropy" ne marche pas !! 
ANN.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


get_ipython().run_cell_magic('time', '', 'ANN.fit(X_train,y_train,epochs=500,batch_size=50,validation_split=0.1)\n')


# sauvegarder le modèle entrainé
pickle.dump(ANN,open('ANN.pickle','wb'))


get_ipython().run_cell_magic('time', '', "test_pred = ANN.predict(X_test)  # => dim (5160, 2) où n =dim[0]de X_test et 2 = dim units dense_last\ny_pred_class = np.argmax(test_pred,axis=1) # on prend l'index du max de chaque ligne prédite\ny_test_class = y_test  # autre nom a y_test pour la suite classification_report et confusion_matrix\n\nprint(classification_report(y_test_class,y_pred_class)) \nprint(confusion_matrix(y_test_class,y_pred_class))\n")


# # Prédiction 

# modèle à choisir
#GBC_3
#SVM
#ANN

text = "produit cassé, livraison lente, pas de réponse de service client"


# # Fonction : text préparation et prédiction 

import spacy
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
    df_stop_word_xls = pd.read_excel('liste_no-stop-words_tokens_unique.xlsx', header=None)
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


# # Modèle 7 - Réseau Neurone

# chargement du bon csv des données en français 
df = pd.read_csv('review_trust_fr_lemmantiser_word+2_VF.csv', sep=',') 
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
