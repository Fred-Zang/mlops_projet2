
import streamlit as st
#import seaborn as sns
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import webbrowser

import sys
import re
import spacy
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # pour modele ANN.h5
nltk.download('punkt')                  # pour modele ANN.h5

from PIL import Image

import pickle
import joblib
#from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf

# --------------------chargement des modèles--------------------------#

nlp = spacy.load('fr_core_news_md') 
# nlp = pickle.load(open("/airflow/clean_model/nlp_SAV_core_news_md.pickle","rb"))

# load Vectorizer For Gender Prediction                                           /airflow/clean_model/vectoriser_ber
# ---------  chargement des modèles et dataframes diverses ------------- #
GBC_2_vectorizer = pickle.load(open("/airflow/clean_model/vectoriser_GBC_2-sav_sklearn102","rb"))
                                    # airflow/clean_model/vectoriser_GBC_2-sav_sklearn102

# load pre-trained model
trained = pickle.load(open(r"/airflow/clean_model/trained.pickle","rb"))

# load Model For Prediction
GBC_2_model = pickle.load(open("/airflow/clean_model/GBC_2-sav_sklearn102.pickle","rb"))

SVM_model = pickle.load(open("/airflow/clean_model/SVM_sav-sklearn102.pickle","rb"))


#---------- Chargement des data csv ----------------------------------#


df_origin = pd.read_csv('/airflow/clean_data/reviews_trust.csv')
df = pd.read_csv('/airflow/clean_data/reviews_trust_fr_VF.csv',index_col = 0) 

data = pd.read_csv('/airflow/clean_data/review_trust_fr_lemmantiser_word+2_VF.csv', sep=',', index_col = 0)

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



# ---------  Fonction de traitement des corpus ------------- #
@st.cache
def text_tokeniser(my_text):
	nlp = spacy.load('fr_core_news_md')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

def text_tokeniser2(my_text):
	nlp = spacy.load('fr_core_news_md')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	
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

# ---------  Fonction d'affichage DataViz ------------- #
def chart(chart):
	if chart == 'Distribution Star variable cible':
		st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
    		
	elif chart == 'Distribution star avec filtre reponse':
		st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
		if st.checkbox("Afficher l'analyse"):
			st.write("Nous remarquons que les sites marchands donnent peu de réponses au clients mécontents ou satisfaits à l'exception des clients très satisfaits, ce qui est anormal et doit être signalé au service marketing.:pray:")
			st.write(" ")
			st.markdown("""
			### Établissons le classement des notes par étoiles suivant :

			- 1 étoile = client très mécontent
			- 2 étoiles = client peu content
			- 3 étoiles = client moyennement satisfait
			- 4 étoiles = client satisfait
			- 5 étoiles = client très satisfait	""")
	elif chart == 'Distribution source avec filtre star':
		st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
		if st.checkbox("Afficher l'analyse"):
			st.markdown("""
			### Variation des données entre ces 2 entreprises

			- TrustedShop a beaucoup + de données 'star' récoltées que son concurrent
			- TrustPilot a essentiellement des notes de clients très mécontents
			- TrustedShop a une grande majorité de clients très satisfaits, jusqu'aux clients peu contents
			""")
    			
	elif chart == 'Distributions des sites marchands de compagny avec filtre star ou filtre source':
		st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
		if st.checkbox("Afficher l'analyse"):
			st.markdown("""
			### Site marchand ***showroom***

			- il a la grande majorité de notations étoiles
			- il a une grande majorité de clients très satisfaits et satisfaits, avec cependant un nombre non négligeable de clients moyennement satisfaits à très mécontents
			- les sources de notations proviennent des 2 entreprises 'TrustPilot' et 'TrustedShop' avec une grande majorité pour ce dernier
			""")
			st.markdown("""
			### Site marchand ***VeePee***

			- il a peu de notations comparé à son concurrent 'showRoom'
			- la grande majorité des notations concerne les clients très mécontents (👉 voir marketing service )
			- l'entreprise 'VeePee' est la seule source de données de notations
			""")
    			
	elif chart == 'Chronologie des notations par Années':
		st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
		if st.checkbox("Afficher l'analyse"):
			st.markdown("""
			- Il y a 66.34% de NaN sur date_commande, le graphe présente essentiellement les informations en 2020 et 2021
			- A partir de 2020, il y a une explosion des commandes, avec une majorité relative de notations positives et cependant bon nombre de clients très mécontents
			""")
	elif chart == ' ':
		st.write(' ')



# ---------  Fonction de préparation des données: filter, token, lemm ------------- #

def present_part1(choose_part): 
	
	if choose_part == 'Présentation Technique du Filtrage et Stop-Words':
		st.write('### :sparkles: Présentation technique ')

		st.write("- L'analyse des sentiment est un problème d'analyse des langages naturels. Mais le *'Machine learning'* modèle n'accepte pas les langages naturels. Il faut numériser les entrées (mots) au préalable.")

		st.write('- Le traitement de langage passe par deux étapes principales :')

		st.write(':a:) Filtrer les textes')

		st.write("##### 	:dart:Objectif:")
		st.write("Afin d'enlever les informations inutiles et réduire la taille des données traitées, un filtrage est nécessaire sur:")

            
		if st.checkbox("Regex = Traitement des caractères ou des mots à conserver ou modifier par une Expression Régulière"):
                
			st.write("Les Expressions régulières servent à traiter le corpus de texte dans de très nombreux cas :")
			st.write("- transformer tout le corpus en lettres minuscules")
			st.write("- éliminer les caractères particuliers comme les chiffres, smileys ou les ponctuations")
			st.write("- faire des recherches de mots, groupes de lettres ou de mots pour les modifier ensuite")
			st.write("- enfin pour de multiples autres applications, si vous voulez + d'info, cliquez sur le lien 'Aide @ Regex'")

			col1, col2, col3, col4, col5 = st.columns(5)
			with col1:
				st.write(' ')
			with col2:
				st.write(' ')
			with col3:
				st.write(' ')
			with col4:
				st.write(' ')
			with col5:
				if st.button('Aide @ Regex'):
					url1 = "https://fr.wikipedia.org/wiki/Expression_r%C3%A9guli%C3%A8re"
					webbrowser.open_new_tab(url1)

 ## 🛑🛑check mots corbeilles ou mot corbeille ? 🛑🛑           
		if st.checkbox("📎 Mots corbeilles (stop words) = Mots inutiles à éliminer"):
    			
			st.write("Il faut bien enlever les mots corbeilles , car ces mots n'ont pas des sens dans l'analyse. Par exemple: le, la, un, des etc.")
			st.write("Selon le retour d'expérience de l'anayse du projet, une liste de mots corbeilles personnalisé est fait. (Voir le fiche *liste_no-stop-words_tokens_unique.xlsx*)")

			st.write("Voici notre procédure pour choix des stop words")
			st.markdown("""
				:one:. Tokenizer les commentaires 👉 
				:two:. Lemmaniser les tokens 👉
				:three:. Compter des tokens lemmanisés par fréquences 👉
				:four:. Séparer les tokens/mots qui **apparaissent seulement 1 fois** et que nous choisissons d'éliminer 👉
				:five:. Sauvegarder ces mots corbeille personnalisés dans un fichier excel 👉 
				:six:. Enlever les mots corbeille *(agrandissez le document ci-dessous pour observer la fréquence des tokens du dataset)*
                    """)

				# 🌞🌞🌞@fred: ajouter le jpd dans le meme folder 🌞🌞🌞
			insert_img('/airflow/data_others/JPG-PNG/sample-image.jpg')  # mots_vides.jpg
		st.write('------------------------------------------------------------------------------')

# Mot particulier
    		#st.write(":paperclip: Mots particulers")
		if st.checkbox("📎 Mots particulers = Mots importants à traiter"):
    			
			
			st.write("Il faut aussi traiter les mots particuliers dans les données. Dans le projet SatisPy, il y a des mots comme 'ras, rad, ras le bol'. Les mots 'ras' et 'rad' présentent 'rien a signaler' ou 'rien à dire', ces sentiments sont est plustôt positifs :smiley:. Cependant le mot 'ras le bol' présente un sentiment plustôt négatif :angry:")
    			# 🌞🌞🌞@fred: ajouter le jpd dans le meme folder 🌞🌞🌞
			 ## 🛑🛑add image apres la recevoir 🛑🛑
			st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
			

			st.write("Notre décision : '")
			st.write(":mega: 'ras, r.a.s, ràs, r.à.s, rad, ràd,r.a.d, r.à.d, r a s, r à s, r a d, r à d' 👉 sont transformés en 'bien'")
			st.write(":mega: 'ras le bol' 👉 est transformé en 'mauvais'")

			st.write("📜: il y a aussi des procédures de **tokenization, Lemmatisation et Vectorisation** à la demande en entrant votre propre message *(Modifiez votre choix en haut de page pour y accéder)*")



	if choose_part == 'Tokeniser, Lemmatiser ou Vectoriser à la demande':
		st.write(':b:) Transformations et Vectorisations')
		st.write('### :sparkles: Comprendre la Tokenisation, Lemmantization et Vectorisation ')
    
		st.write("Tokeniser, Lemmantiser et Vectoriser une phrase en direct, à vous de jouer ! 🎈")
		st.write("- Tokeniser  = Transformer le texte en jetons de mots uniques sous forme de liste")
		st.write("- Lemmatiser = Transformer les mots en leur racine sémantique ")
		st.write("- Vectoriser = Transformer le texte en matrice de nombres que le modèle de prédiction pourra utiliser")


		message = st.text_area("Entrer un commentaire puis taper Ctrl + Entrée pour le valider","Ecrire ici ...") 
        
		if st.button("Tokeniser et Lemmatiser votre message"):
			nlp_result = text_tokeniser(message)
			st.json(nlp_result)
                
		if st.button("Vectoriser votre message"):
			vectorizer = CountVectorizer()
			nlp_result = text_tokeniser2(message)
			nlp_result = [' '.join(nlp_result)]
			text_vec = vectorizer.fit_transform(nlp_result).todense()
			mots = vectorizer.get_feature_names()
			st.write('Message vectorisé:')
			st.write(text_vec)
			df_vec = pd.DataFrame(text_vec, columns = mots)
			st.dataframe(df_vec)
			# test phrase : la livraison est rapide. je suis contente. la service est bon.

		col1, col2, col3 = st.columns(3)
		with col1:
			if st.button('Aide @ Tokenizer'):
				url8 = "https://fr.wikipedia.org/wiki/Tokenisation_(s%C3%A9curit%C3%A9_informatique)"
				webbrowser.open_new_tab(url8)
		with col2:
			if st.button('Aide @ Lemmatizer'):
				url9 ="https://fr.wikipedia.org/wiki/Lemmatisation"
				webbrowser.open_new_tab(url9)
		with col3:                             
			if st.button('Aide @ CountVectorizer'):
				url2 = "https://www.youtube.com/watch?v=FwSD1EM2Qkk"
				webbrowser.open_new_tab(url2)




# ---------  Fonction de prédiction à la demande ------------- #

def present_part(choose_do):
	if choose_do == 'Modelisations à la demande':

		st.markdown('# Modelisations à la demande')
		

		corpus = st.text_area("Entrer un commentaire puis taper Ctrl + Entrée pour le valider","Ecrire ici ..")
		corpus = corpus.lower()
		
		models = ['GBC_2','SVM', 'ANN']
		model = st.selectbox(label = 'Choix du modèle', options = models)
		# charger le modele préentraine widipedia en avance: je ne considère pas l'opimisation des temps ici. pas encore, on charge le modèle meme si le GBC_2 n'utilse pas
		

		if model =='GBC_2':
			pred = GBC_2(corpus)
			sentiment(pred) 
		elif model =='SVM':
			pred = SVM(corpus)
			sentiment(pred)
		elif model == 'ANN':
			ANN = tf.keras.models.load_model(r'/airflow/clean_model/ANN-tensor280.h5')  # anciennement ANN.h5
			pred = prediction(corpus, ANN) 
		#st.write(pred)
			pred = pred[0].tolist()
			pred = pred.index(max(pred))
			sentiment(pred)
			st.write(pred)
		else:
			st.write('')
			# je ne sais pas si nous avons besoin d'ajouter autres modèle ou pas 		
	
	else:  
		if st.button('Aide @ Regression Logistique'):
			url3 = "https://fr.wikipedia.org/wiki/R%C3%A9gression_logistique"
			webbrowser.open_new_tab(url3)
		if st.button('Aide Vidéo Sagesse de la Foule'):
			url4 = "https://youtu.be/7C_YpudYtw8"  # sagesse de la foule Guillaume
			st.video(url4)
		if st.button('Aide @ wikipedia2vec'):
			url5 = "https://wikipedia2vec.github.io/wikipedia2vec/intro/"
			webbrowser.open_new_tab(url5)
		if st.button('Aide tuto video wikipedia2vec + modèle'):
			url8 = "https://youtu.be/FwSD1EM2Qkk"
			st.video(url8)
		if st.button('Aide vidéo Tuto modèle pré-entrainé wikipedia2vec'):        
			url6 = "https://www.youtube.com/watch?v=FwSD1EM2Qkk"
			webbrowser.open_new_tab(url6)
		if st.button('Aide Vidéo Perceptron'):
			url7 = "https://youtu.be/VlMm4VZ6lk4"  # perceptron
			st.video(url7)


# ---------  Fonction d'affichage de sentime ------------- #
def sentiment(pred):
	if pred == 1:
		st.subheader("Prédiction = Sentiment Positif:smiley: ")
		#st.write('Votre commentaire est classifié: ', pred)
		#st.write('Merci à votre *positif* commentaire :smile:')
	else:
		st.subheader("Prédiction = Sentiment Négatif :angry: ")
		#st.write('Votre commentaire est classifié: ', pred)
		#st.write('Merci à votre *négative* commentaire :angry:. Pour améliorer notre service, la site vous contactera.')

# ---------  Fonction la prédiction par SVM_wiki ------------- #
def SVM(corpus):
	# 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
	trained = pickle.load(open('/airflow/clean_model/trained.pickle','rb'))
	# charger le model
	# 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
	SVM_wiki = pickle.load(open('/airflow/clean_model/SVM_sav-sklearn102.pickle','rb'))
	text_vec = my_doc2vec(corpus,trained)
	text_vector = pd.DataFrame(my_doc2vec(corpus, trained)).T
	pred = SVM_wiki.predict(text_vector)
	return int(pred)

# ---------  Fonction la prédiction par GBC ------------- #
def GBC_2(corpus):
	# charger le vecteur
	# 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
	vectorizer = pickle.load(open('/airflow/clean_model/vectoriser_GBC_2-sav_sklearn102','rb'))
	# charger le model 
	# 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
	GBC_2 = pickle.load(open('/airflow/clean_model/GBC_2-sav_sklearn102.pickle','rb'))
	text= pd.Series(corpus)
	text_vec = vectorizer.transform(text).todense()
	#st.write('vectorization for GBC2 ',text_vec)
	pred = GBC_2.predict(text_vec)
	pred = int(pred)
	return pred

# ---------  Fonction la prédiction par ANN, GBC_3 et SVM ------------- #

def prediction(text, model):
	# 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
	trained = pickle.load(open('/airflow/clean_model/trained.pickle','rb'))
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
	# 🌞🌞🌞@fred: changer le file path 🌞🌞🌞
	df_stop_word_xls = pd.read_excel('/airflow/clean_data/liste_no-stop-words_tokens_unique.xlsx', header=None)
	update_list_fr = list(df_stop_word_xls[0])
	# initialisation de la variable des mots vides
	stop_words = set()
	stop_words.update(update_list_fr)
	my_doc_sw = stop_words_filtering(my_doc_lem,stop_words)
	my_vec = my_doc2vec(my_doc_sw,trained)
    

	pred_my_doc = model.predict(my_vec.reshape(1,-1))
	return pred_my_doc

# ---------------------- créer 4 pages différentes sur notre streamlit --------------------------------------  #
# créer une liste de 4 noms des pages

pages = [
    "Le Projet en DETAILS",
    "Cahier des Charges",
    "Filtrages, Tokenisations, Lemmatisations et Vectorisations",
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
    # écrire du texte ( # = taille, ##, ###) ici le titre de la page
    st.write("### Présentation Générale !!!:")
    st.write("blablabla....................😉​")
    st.write("### Rapide aperçu du Projet :")
    st.write(
        "On nous a remis un dataset 'reviews_trust.csv' comportant 19.863 lignes et 11 colonnes qui correspond aux commentaires clients \
        et notation de 1 à 5 étoiles sur leurs achats de produit sur 2 sites marchands 'ShowRoom' et 'VeePee'. Ces commentaires proviennent de \
            2 sources récoltant les avis, 'TrustedShop' et 'TrustPilot' et voici un extrait du dataset :")
    st.write("--- AERER TOUT CA ---")
    st.write("Projet présenté et réalisé par Quan Liu, Éric Gasniere et Fred Zanghi")


#------------------------------------------------------------------------------------------
# séparation des pages  
#------------------------------------------------------------------------------------------
elif page==pages[1]:  # sur la page 1 Dataviz
	# affichage
	insert_head('https://datascientest.fr/train/assets/logo_datascientest.png','/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
	
	st.markdown("<h2 style='text-align: center; color: white;'>Cahier des Charges</h2>", unsafe_allow_html=True)
	st.markdown("""
	## Objectif：
	- Exploration des données 
	- Visualisation des données
	- Préparation des données 
	""")

	# afficher les données et la première analyse

	st.markdown('------------------------------------------------------------------------------------------')
	st.markdown('## 🎈Exploration des données')
    
	# 🌞🌞🌞@fred: changer le file path 🌞🌞🌞

	st.write("## Les données originales")
	st.dataframe(df_origin.sample(60))           # ouf j'ai du changer tous les df en df_origin partout ici !

	if st.checkbox("Afficher les valeurs manquantes"): 
		st.dataframe(df_origin.isna().sum())

	# 🛑🛑je déplace ce bloc de code ici car il est lié avec l'exp des données🛑🛑

	st.markdown('## 🎈 Visualisation des données')

	st.markdown("""
	### Objectif：
	- Définition des métriques et exigences de performances
	- Schéma d’implémentation
	- Récupération de nouvelles données

	""")

	charts = ['Distribution Star variable cible',
	'Distribution star avec filtre reponse',
	'Distribution source avec filtre star', 
	'Distributions des sites marchands de compagny avec filtre star ou filtre source', 
	'Chronologie des notations par Années',' ']

	chart_choose = st.selectbox(label = "Choix de chart", options = charts)
    
	#Afficher les charts
	chart(chart_choose)

	st.markdown('## 🎈 Préparation des données')
	if st.checkbox("Afficher les tâches principales"):
		st.markdown("""
		### Tâches principales：
		- Nettoyer des données
		- Traitement des données (commentaires, data, star etc)
		- Sauvegarde des données
		""")

	st.markdown(
	"""### Choix de la langue analysée : Extraire les commentaires en Français
	Il y a plusieurs langues dans les données, afin de modéliser le sentiment.
	Ici le package 'langid' est choisi pour faire la classification des langages. 
	Après la classification, le langage plus fréquent est le Français à 87.1 % 
	""")

	#dic_lang = {'Langage' : ['fr','en','it','es','pt','wa','de','oc','mt','nl'],'%':[87.1,4.2,2.1,2.1,1.9,0.6,0.4,0.2,0.2,0.2]}
	#table_lang = pd.DataFrame(dic_lang)
	#fig = plt.figure(figsize = (8,4)) 
	#sns.barplot(x = 'Langage',y = '%', data = table_lang);
	#st.pyplot(fig)
	st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')

	st.markdown('------------------------------------------------------------------------------------------')

	st.markdown('## 🎈 WordCloud')
	if st.checkbox("Afficher la description du Wordcloud"):
		st.markdown("""
		### WordCloud：
		Le nuage de mots (WordCloud) est une représentation visuelle qui complète une section de texte pour aider les lecteurs à mieux visualiser la présence de mots clés du texte concerné
		La bibliothèque wordcloud implémente un algorithme permettant d'afficher un nuage de mots d'un texte. Cet algorithme regroupe les étapes suivantes :
		- Tokeniser le texte passé en paramètre
		- Filtrer les mots vides
		- Calculer la fréquence des mots
		- Représenter visuellement les mots-clefs les plus fréquents sous forme de nuage de mots (source: https://datascientest.com/)
		""")
 
	# add wordcloud
	st.write('### Wordcloud du dataset original avant traitements et filtrages')
	#☻ afficher l'image wordcloud
	st.image('/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')

#------------------------------------------------------------------------------------------
# sépration des pages  
#------------------------------------------------------------------------------------------
# sur la page 2 Vectorisation & Filtrage 
elif page ==pages[2]:
	
	# affichage
	insert_head('https://datascientest.fr/train/assets/logo_datascientest.png','/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
	# title du page
	st.markdown("<h1 style='text-align: center; color: white;'>Filtrages, Tokenisation, Lemmatisations et Vectorisations</h1>", unsafe_allow_html=True)
	st.markdown("""
	### Objectif：
	- Vectorisation & Filtrage des Commentaires
	- Démonstration 
	""")

	st.write("	:bookmark: Le sentiment est classifié comme 'positif' *1*:smiley: ou 'negatif' *0*:angry:")
	st.write("Voici le tableau entre le sentiment (*1* ou *0*) et les notes correspondantes")
	df = pd.DataFrame({'Sentiment': [0,1],'Note':[[1,2],[3,4,5]]},index = ['négatif','positif'])

	st.dataframe(df)
	choose = ['Présentation Technique du Filtrage et Stop-Words', 'Tokeniser, Lemmatiser ou Vectoriser à la demande']

	choose_part = st.selectbox(label = '🎈 CHOISIR ENTRE PRÉSENTATION TECHNIQUE OU DÉMONSTRATION PAR VOUS-MÊME 🎈', options  = choose)
	
	present_part1(choose_part)

#------------------------------------------------------------------------------------------
# sépration des pages  
#------------------------------------------------------------------------------------------
elif page ==pages[3]:  # sur la page 3 Modelisation
    
    #############  TOUS LES MODELES ET DATA ICI SONT RECUPERES DANS /airflow/clean_data ou /airflow/clean_model #############################
    
	# affichage
	insert_head('https://datascientest.fr/train/assets/logo_datascientest.png','/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
	
	st.markdown("<h2 style='text-align: center; color: white;'>Nos 3 Modèles présentés ici</h2>", unsafe_allow_html=True)
	

	# title du page
	st.markdown("<h1 style='text-align: center; color: white;'>Modélisations à la demande</h1>", unsafe_allow_html=True)

	#st.write("# Modélisations à la demande")
	message = st.text_area("Entrer un commentaire puis taper Ctrl + Entrée pour le valider","Ecrire ici ..")  
	corpus = message.lower()
	if st.button("Gradient Boosting Classifier 2"):
		pred = GBC_2(corpus)
# 🛑🛑mélange anglais et français ??🛑🛑
		st.success('Your message was classified as {}'.format(pred))
		sentiment = sentiment(pred)
		st.write("Ce modèle atteint une précision de 89% sur les 2 sentiments et donc malgré tout un taux d'erreurs de 11%")


	if st.button("pré-entrainement avec Wikipedia2vec puis modélisation par SVM"):
		pred = SVM(corpus)
# 🛑🛑mélange anglais et français ??🛑🛑
		st.success('Your message was classified as {}'.format(pred))
		sentiment(pred)
		st.write("Ce modèle atteint une précision de 90% sur les 'satisfaits' et seulement 76% sur les 'mécontents")


	if st.button("ANN Réseaux de Neurones Articiciels"):
		ANN = tf.keras.models.load_model('/airflow/clean_model/ANN-tensor280.h5')
		pred = prediction(corpus, ANN) 
		st.write(pred)
		pred = pred[0].tolist()
		pred = pred.index(max(pred))
		sentiment(pred)
		st.write('')
	st.write("Ce modèle atteint une précision de 86% sur les 'satisfaits' et seulement 73% sur les 'mécontents")
# 🛑🛑ajouter les image dans le dossier et décommenté🛑🛑
	#	st.image("sample-image.jpg")  # "ANN-layers_fit_confusion_classifreport.jpg")
	#	#st.write('add img')
	#	st.write("La fonction Loss (coût des erreurs à minimiser) termine à 29.9% ce qui n'est pas optimum ")

#------------------------------------------------------------------------------------------
# sépration des pages  
#------------------------------------------------------------------------------------------

elif page ==pages[4]:  # sur la page 4 Conclusion
	# affichage
	insert_head('https://datascientest.fr/train/assets/logo_datascientest.png','/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png')
    
	# title du page
	st.markdown("<h1 style='text-align: center; color: white;'>✨ Conclusion ✨</h1>", unsafe_allow_html=True)
	st.write("")
	col1, col2, col3 = st.columns([1,3,1])
	with col2:
		st.write("Traiter un sujet comme celui-ci a été passionnant et nous aimerions le poursuivre encore")
		st.write("")
		st.write("Nous y avons mis beaucoup de cœur, à découvrir toutes ces notions à travers nos modules de cours et à dérouler une grande quantité de modèles pour trouver des approches et ressources différentes.")
		st.write("")
		st.write("Nous avons le sentiment d’en être qu’au tout début de notre étude, tant il nous reste de questions à résoudre, de tests à réaliser et de nouvelles pistes à explorer.")
		st.write("")
		st.write("Nous espérons que sa lecture vous a été agréable et vous recommandons de parcourir les vidéos et liens donnés en annexe si vous souhaitez en savoir plus sur le sujet.")
		st.write("")
		st.markdown("<h1 style='text-align: center; color: white;'>🌼​ Remerciements 🌼​</h1>", unsafe_allow_html=True)
		st.write("")
		st.write("Nous tenons à remercier toute l’équipe de DataScientest pour son écoute et leurs conseils, et particulièrement notre mentor de projet Antoine qui nous a suivi chaque semaine en réunion zoom avec un sourire et une patience admirable.")
  
	    