#################################################################
#    MODELES RETENUS POUR L'ANALYSE DE SENTIMENT                #
#################################################################

#####################################################
#                                                   #
#    1. MODELE GBC - GRADIENT BOOSTING CLASSIFIER   #
#                                                   #
#    2.1 PASSAGE DU CORPUS SUR UN MODELE            #
#            PRE-ENTRAINE EN LANGUE FRANCAISE       #
#            PROVENANT DE WIKIPEDIA                 #
#            DE 2.500.733 MOTS VECTORISES           #
#                                                   #
#    2.2 MODELE SVM - SUPPORT VECTOR MACHINES       #
#                                                   #
#    2.3 MODELE ANN - ARTIFICIAL NEURAL NETWORK     #
#                                                   #
#####################################################


# CHARGEMENT DES PACKAGES
import datetime
from my_functions import transform_star_to_target, collect_stopwords, prepare_data, save_report
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks

import sys
sys.path.append('/app/clean_functions')


#####################################################
#   1. MODELE GBC - GRADIENT BOOSTING CLASSIFIER    #
#####################################################
def GBC_predict_df(
        path_data,
        path_stopwords='/app/clean_data/liste_no-stop-words_tokens_unique.xlsx',
        save_model=True,
        path_save_model='/app/clean_model/GBC_2-sav_DAG.pickle',
        save_vectorizer=True,
        path_save_vectorizer='/app/clean_model/vectoriser_GBC_2-sav_DAG'):
    """Prédiction avec un Gradient Boosting Classifier:
            1.1 Préparation des données
            1.2 Entrainement du modèle
            1.3 Sauvegarde du modèle
            1.4 Evaluation du modèle

    Parameters:
    -----------
        path_data: Chemin de stockage des données nettoyées
        path_stopwords: Chemin du fichier contenant les stopwords
        save_model: Boolean indiquant si on sauve le modèle ou pas
        path_save_model: Chemin de sauvegarde du modèle GBC
        save_vectorizer: Boolean indiquant si on sauve le vectorizer ou pas
        path_save_vectorizer: Chemin de stockage du vectorizer

    Returns:
    --------
        les prédictions
    """

    # 1.1 PREPARATION DES DONNES
    # ------------------------------------
    # Importer les données nettoyées
    df = pd.read_csv(path_data, index_col=0)

    # Récupération des features (colonne de commentaires)
    features = df.Commentaire

    # Récupération des target
    #   1 ou 2 étoiles    => Sentiment négatif (0)
    #   3, 4 ou 5 étoiles => Sentiment positif (1)
    target = transform_star_to_target(df)

    # Séparation du jeu de données en jeu d'entrainement et de test
    X_train_GBC_2, X_test_GBC_2, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=20)

    # Récupération des mots vides (ou stop words)
    stop_words = collect_stopwords(path_stopwords)

    # Initialiser un objet vectorizer, garde aussi un mot avec plus 2 lettres
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"[a-zéèêàâîïàùçôëûæœ]{2,}")

    # Mettre à jour la valeur de X_train et X_test
    X_train_GBC_2 = vectorizer.fit_transform(X_train_GBC_2).todense()
    X_test_GBC_2 = vectorizer.transform(X_test_GBC_2).todense()

    # print(vectorizer.vocabulary_)

    # 1.2 ENTRAINEMENT DU MODELE
    # ------------------------------------
    GBC_2 = GradientBoostingClassifier(  # The number of estimators as selected by early stopping
        n_estimators=100,
        # Learning rate shrinks the contribution
        # of each tree by learning_rate
        learning_rate=1.0,
        # Maximum depth of the individual regression estimators.
        # Controls the random seed given to each
        # Tree estimator at each boosting iteration
        max_depth=1,
        random_state=0
    ).fit(X_train_GBC_2, y_train)

    # 1.3 SAUVEGARDE DU MODELE
    # ----------------------------------
    dt = datetime.datetime.now()
    time = dt.strftime('%Y-%m-%d-%H-%M')
    # sauvegarder le modèle pré-entrainé
    if save_model:
        pickle.dump(GBC_2, open(path_save_model, 'wb'))
        part1, part2 = path_save_model.split('.')
        pickle.dump(GBC_2, open(part1 + '_' + time + '.' + part2, 'wb'))

    # sauvegarder le Vectoriser
    if save_vectorizer:
        pickle.dump(vectorizer, open(path_save_vectorizer, 'wb'))
        pickle.dump(vectorizer, open(path_save_vectorizer + '_' + time, 'wb'))

    # 1.4 EVALUATION DU MODELE
    # ----------------------------------
    # Calculer les prédictions
    y_pred_GBC_2 = GBC_2.predict(X_test_GBC_2)

    # REPORTING - PERFORMANCES - SAUVEGARDE DANS UN FICHIER AVEC HISTORIQUE (f1-score, accuracy)
    # Collecte du nouveau rapport de classification au format dict
    classif = classification_report(y_test, y_pred_GBC_2, output_dict=True)
    save_report(
        classif,
        time,
        data_store_path='/app/clean_data/',
        data_name='reporting_GBC2')


#####################################################
#       2.1 PASSAGE DU CORPUS SUR UN MODELE         #
#            PRE-ENTRAINE EN LANGUE FRANCAISE       #
#            PROVENANT DE WIKIPEDIA                 #
#            DE 2.500.733 MOTS VECTORISES           #
#####################################################

# Chargement du modèle pré-entrainé de wikipedia2vec
# - TUTO d'aide pour charger une réprésentation pré-entraînée dans une langue souhaitée de wikipedia2vec
#
#    - 1 aller sur le lien @  https://wikipedia2vec.github.io/wikipedia2vec/pretrained/#french
#    - 2 copier le lien d'adresse d'un des 4 fichiers souhaité en langue Française de la ligne
#      frwiki_20180420 (window=5, iteration=10, negative=15): 100d (bin) 100d (txt) 300d (bin) 300d (txt)
#    - 3 placer le lien sur notre IDE spyder ou jupyter
#    - 4 avec la souris sur ce chemin, faire CRL + clic
#               => le fichier se télécharge = 1h pour le + petit de 450Mo nommé 100d(txt)
#    - 5 placer ce fichier dans un dossier et le décompresser => 1.9Go
#    - 6 placer ce fichier décompressé dans le dossier
#           C:\\Users\\UserPC\\anaconda3\\envs\\spyder-env\\lib\\site-packages\\gensim\\test\\test_data\\
#      où spyder-env est mon environnement spécial spyder (pas obligatoire)
#    - 7 enfin lancer la cellule ci-dessous pour charger le fichier sur notre IDE et créer un vecteur
#
# Le chargement d'un objet gensim prend 6 à 8 mins chaque fois, le modèle **trained** est sauvegardé et utilisé dans la suite
#
# si c'est la première fois et que vous n'avez pas récupéré le
# trained.pickle, lancez les codes. sinon charger directement le modèle
'''
# chargement d'un objet gensim de type models.keyedvectors de taille 2.500.733
start = time()

trained = load_word2vec_format('frwiki_20180420_100d.txt.bz2', binary=False)    # 6 à 8 min de chargement, patience...

end = time()
print("temps chargement modele pré-entrainé = ", end-start)
# à voir absolument le parametre unicode_errors='ignore' ou 'replace' qui sera utile par la suite
# pour éviter des ereurs en chargement

print(trained.vectors.shape)
'''
'''
# sauvegarder le modèle pré-entrainé  trained.pickle
'''

#####################################################
#    2.2 MODELE SVM - SUPPORT VECTOR MACHINES       #
#####################################################


def SVM_predict_df(
        path_data='/app/clean_data/review_trust_fr_lemmantiser_word+2_VF.csv',
        path_model_wiki='/app/clean_model/trained.pickle',
        save_model=True,
        path_save_model='/app/clean_model/SVM_sav-DAG.pickle'):
    """Prédiction avec un Support Vector Machines:
            2.2.1 Préparation des données
            2.2.2 Entrainement du modèle
            2.2.3 Sauvegarde du modèle
            2.2.4 Evaluation du modèle

    Parameters:
    -----------
        path_data: Chemin de stockage des données nettoyées
        path_model_wiki: Chemin du modèle Wikipedia
        save_model: Boolean indiquant si on sauve le modèle ou pas
        path_save_model: Chemin de sauvegarde du modèle SVM

    Returns:
    --------
        les prédictions
    """
    # 2.2.1 PRETRAITEMENT DU CORPUS ET CREATION DES JEUX D'ENTRAINEMENT ET
    # ------------------------------------
    X_train, X_test, y_train, y_test = prepare_data(path_data, path_model_wiki)

    # 2.2.2 ENTRAINEMENT DU MODELE
    # ------------------------------------
    # SVM avec noyau RBF par défaut
    SVM = SVC(C=100, gamma=1, random_state=0)
    # Ces paramètres ont été trouvés par GridSearchCV sur quelques valeurs seulement
    # Nous donnons le code pour cela ci-dessous mais à lancer avec parcimonie
    # puisque les temps de calculs ont sont multipliés !
    SVM.fit(X_train, y_train)

    # RECHERCHE DES MEILLEURS PARAMETRES

    # recherche des best parametres avec gridsearshCV
    # from sklearn.model_selection import GridSearchCV

    # param_grid = {'C':[50, 100], 'kernel': ['rbf'], 'gamma': [1, 10]}  # recherche des bons hyperparams
    # grid = GridSearchCV(SVC(), param_grid) # création de la grille de
    # recherche sur le model SVC()

    # grille = grid.fit(X_train, y_train)
    # affichage des résultats test de GridSearchCV
    # print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']])
    # print("grid.best_params :", grid.best_params_)
    # svm_best= grid.best_estimator_

    # 2.2.3 SAUVEGARDE DU MODELE
    # ------------------------------------
    dt = datetime.datetime.now()
    time = dt.strftime('%Y-%m-%d-%H-%M')
    if save_model:
        pickle.dump(SVM, open(path_save_model, 'wb'))
        part1, part2 = path_save_model.split('.')
        pickle.dump(SVM, open(part1 + '_' + time + '.' + part2, 'wb'))

    # 2.2.4 EVALUATION DU MODELE
    # ------------------------------------
    # Calcul des prédictions
    y_pred_SVM = SVM.predict(X_test)

    # REPORTING - PERFORMANCES - SAUVEGARDE DANS UN FICHIER AVEC HISTORIQUE
    # (f1-score, accuracy)
    classif = classification_report(y_test, y_pred_SVM, output_dict=True)
    save_report(
        classif,
        time,
        data_store_path='/app/clean_data/',
        data_name='reporting_SVM')


#####################################################
#    2.3 MODELE ANN ARTIFICEL NEURONAL NETWORK      #
#####################################################
def ANN_predict_df(path_data,
                   path_model_wiki='/app/clean_model/trained.pickle',
                   save_model=True,
                   path_save_model='/app/clean_model/ANN-sav-DAG.h5',
                   ):
    """Prédiction avec un réseau de neurones:
            2.3.1 Préparation des données
            2.3.2 Création du modèle
            2.3.3 Entrainement du modèle
            2.3.4 Sauvegarde du modèle
            2.3.5 Evaluation du modèle

    Parameters:
    -----------
        path_data: Chemin de stockage des données nettoyées
        path_model_wiki: Chemin du modèle Wikipedia
        save_model: Boolean indiquant si on sauve le modèle ou pas
        path_save_model: Chemin de sauvegarde du modèle ANN

    Returns:
    --------
        les prédictions
    """
    # 2.2.1 PRETRAITEMENT DU CORPUS ET CREATION DES JEUX D'ENTRAINEMENT ET
    # ------------------------------------
    X_train, X_test, y_train, y_test = prepare_data(path_data, path_model_wiki)

    # 2.2.2 CREATION DU MODELE
    # ------------------------------------
    inputs = Input(shape=(100))
    dense_1 = Dense(units=64, activation="relu")
    dropout_2 = Dropout(rate=0.1)
    dense_3 = Dense(units=32, activation="relu")
    dropout_4 = Dropout(rate=0.1)
    dense_5 = Dense(units=2, activation="sigmoid")

    x = dense_1(inputs)  # -> x de type engine.keras_tensor.KerasTensor
    x = dropout_2(x)
    x = dense_3(x)
    x = dropout_4(x)
    # tous les x suivant et le outputs sont du meme type que le 1er x
    outputs = dense_5(x)
    # - Les commandes suivantes permettent de finaliser la définition du modèle et d'en afficher la structure.
    ANN = Model(inputs=inputs, outputs=outputs, name="ANN")
    ANN.summary()

    # Compilation du modele avec "binary_crossentropy" ne marche pas !!
    ANN.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

    # AUTOMATIC REDUCTION OF THE LEARNING RATE
    lr_plateau = callbacks.ReduceLROnPlateau(  # Metric to control
        monitor='val_loss',
        # Number of epochs to wait before
        # decreasing the learning rate
        patience=5,
        # Percentage learning rate is decreased
        factor=0.1,
        # Number of informations displayed during training
        verbose=2,
        # Indicate the metric decrease
        mode='min')

    # EARLY STOPPING
    early_stopping = callbacks.EarlyStopping(  # Metric to control
        monitor='val_loss',
        # Number of epochs to wait before
        # stopping the training
        patience=13,
        # Indicate the metric decrease
        mode='min',
        # To restore the weights of the best epoch
        restore_best_weights=True)

    # 2.2.3 ENTRAINEMENT DU MODELE
    # ------------------------------------
    BATCH_SIZE = 128

    ANN.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            # Number of epochs
            epochs=100,
            # Number of steps per Epoch
            steps_per_epoch=(len(X_train)) // (BATCH_SIZE * 2),
            # Number of iterations during the test
            validation_steps=(len(X_test)) // (BATCH_SIZE * 2),
            # Callbacks
            callbacks=[
                lr_plateau,
                early_stopping],
            # Using all processors
            workers=-1)

    # 2.2.4 SAUVEGARDE DU MODELE
    # ------------------------------------
    dt = datetime.datetime.now()
    time = dt.strftime('%Y-%m-%d-%H-%M')
    if save_model:
        ANN.save(path_save_model)
        part1, part2 = path_save_model.split('.')
        ANN.save(part1 + '_' + time + '.' + part2)

    # 2.2.5 EVALUATION DU MODELE
    # ------------------------------------
    # Calcul des prédictions
    # => dim (5160, 2) où n =dim[0]de X_test et 2 = dim units dense_last
    test_pred = ANN.predict(X_test)
    # on prend l'index du max de chaque ligne prédite
    y_pred_ANN = np.argmax(test_pred, axis=1)
    # y_pred_ANN = np.where(test_pred[:,1] > 0.52, 1, 0)

    # REPORTING - PERFORMANCES - SAUVEGARDE DANS UN FICHIER AVEC HISTORIQUE
    # (f1-score, accuracy)
    classif = classification_report(y_test, y_pred_ANN, output_dict=True)
    save_report(
        classif,
        time,
        data_store_path='/app/clean_data/',
        data_name='reporting_ANN')
