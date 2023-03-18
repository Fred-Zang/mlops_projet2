# CHARGEMENT DES PACKAGES
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# PRETRAITEMENT DES DONNEES - TRANSFORMATION DES ETOILES EN CIBLE (0 ou 1)
def transform_star_to_target(df_comment):
    """Transforme les étoiles en cible:
            1 ou 2 étoiles    => Sentiment négatif (0)
            3, 4 ou 5 étoiles => Sentiment positif (1)

    Parameters:
    -----------
        df_comment : DataFrame contenant les commentaires (str)

    Returns:
    --------
        Liste contenant des entiers (0 ou 1)
    """
    new_star = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    df_comment.star.replace(new_star, inplace=True)
    return df_comment.star.astype('int')


# TRAITEMENT DES MOTS VIDES (ou STOP WORDS)
def collect_stopwords(path):
    """Collecte des stopwords stockés dans un fichier Excel

    Parameters:
    -----------
        path : Chemin pour accéder au fichier Excel

    Returns:
    --------
        set de mots vides (ou stop words)
        (a set is a collection of unique data)
    """
    # Chargement fichier excel de stop words french en dataframe
    df_stop_word_xls = pd.read_excel(path, header=None)

    # Création de stop words set
    # Liste selon le retour d'expérience projet
    update_list_fr = list(df_stop_word_xls[0])

    # Initialisation de la variable des mots vides
    stop_words = set()
    stop_words.update(update_list_fr)
    return stop_words


# REPORTING - PERFORMANCES
def reporting(modele, y_test, y_pred):
    """Calcul et affichage du rapport de classification et de la matrice de confusion d'un modèle

    Parameters:
    -----------
    modele : str
        Nom du modèle testé
    y_test : DataFrame contenant des entiers (0 => sentiment négatif, 1 => sentiment positif)
        Target des données de test
    y_pred : DataFrame contenant des entiers (0 => sentiment négatif, 1 => sentiment positif)
        Prédiction

    Returns:
    --------
    La méthode ne retourne rien. Elle affiche les rapports.
    """
    # Calcul et affichage de classification_report
    print("PERFORMANCES DU MODELE {}".format(modele))
    print("RAPPORT DE CLASSIFICATION DU MODELE")
    print(classification_report(y_test, y_pred))
    print()

    # Calcul et affichage de la matrice de confusion
    confusion_matrix = pd.crosstab(
        y_test,
        y_pred,
        rownames=['Classe réelle'],
        colnames=['Classe prédite'])
    print("MATRICE DE CONFUSION")
    print(confusion_matrix)
    print()


# MODELE PREENTRAINE AVEC WIKIPEDIA - SIMILARITES DE MOTS
def find_similar(path_model_wiki, positive, negative):
    """_summary_

    Args:
    -----
        path_model_wiki (str): Chemin du modèle Wikipedia ('../data/Modeles/trained.pickle')
        positive (list of str): Liste de mots (Ex: ['roi', 'femme'])
        negative (list of str): Liste de mots (Ex: ['homme'])

    Returns:
    --------
        Le mot le plus probable avec les similarités positives et négatives (Ex: 'reine')
    """
    # charger le modèle pré-entrainé sauvegardé
    trained = pickle.load(open(path_model_wiki, 'rb'))

    # exemple commun de test calculatoire sur les token : +A - C => +B - ?
    # avec A=roi, C = homme, B=femme => ? = reine solution
    return trained.most_similar(positive=positive, negative=negative)[0][0]


# FONCTION my_doc2vec() DE PRETRAITEMENT
def my_doc2vec(doc, trained):
    """Fonction pour transformer un document (ici une ligne de commentaire) en vecteur à partir des tokens qui le compose

    Parameters:
    -----------
        doc (str): lignes de commentaires à traiter
        trained (.pickle): modèle pré-entrainé sur wikipedia

    Returns:
        Vecteur représentant le document (vecteur nul si aucun token trouvé)
    """

    # Dimension de représentation
    p = trained.vectors.shape[1]   # p = 100
    # Intitialisation du vecteur
    vec = np.zeros(p)  # array (100,)
    # Initialisation du nombre de tokens trouvés à 0
    nb = 0
    # Traitement de chaque token de la ligne de commentaire
    for tk in doc:
        # ne traiter que les tokens reconnus
        try:
            values = trained[tk]  # on récupère le vecteur du token concerné
            vec = vec + values   # on incrémente ce vecteur dans vec
            nb = nb + 1.0        # le compteur de token d'incrémente
        except BaseException:
            pass  # pour lever l'erreur si aucun token trouvé dans notre modele pré-entrainé
        # moyenne des valeurs uniquement si on a des tokens reconnus
    if (nb > 0.0):
        vec = vec / nb
    return vec  # renvoie le vecteur moyenné ou un vecteur nul si aucun token trouvé


# PRETRAITEMENT DU CORPUS ET CREATION DES JEUX D'ENTRAINEMENT ET DE TEST
# POUR LES MODELES SVM ET ANN
def prepare_data(path_data, path_model_wiki):
    """Prétraitement du corpus afin de fournir les jeux d'entrainement et de test

    Args:
    -----
        path_data (str): Chemin de stockage des données nétoyées ('../data/review_trust_fr_lemmantiser_word+2_VF.csv')
        path_model_wiki (str): Chemin du modèle Wikipedia ('../data/Modeles/trained.pickle')

    Returns:
    --------
        X_train, X_test, y_train, y_test: Jeux d'entrainement et de test
    """

    # Charger le modèle pré-entrainé sauvegardé
    trained = pickle.load(open(path_model_wiki, 'rb'))

    # Importer les données nettoyées
    data = pd.read_csv(path_data, sep=',')
    data.drop(data.iloc[:, :1], axis=1, inplace=True)
    # data.sample(1)

    # Préparer la liste corpus_sw pour modèle
    corpus_sw = data['no_stop_words'].tolist()
    # corpus_sw = data['words+3'].tolist()

    # Traiter les documents du corpus
    docsVec = list()
    # Pour chaque ligne du corpus nettoyé
    for doc in corpus_sw:
        # calculer son vecteur
        vec = my_doc2vec(doc, trained)
        # ajouter dans la liste
        docsVec.append(vec)

    # Transformer en array numpy
    matVec = np.array(docsVec)

    # Transformer en dataframe
    X = pd.DataFrame(matVec, columns=["v" + str(i + 1)
                     for i in range(matVec.shape[1])])

    # Récupération de la cible
    y = data.star

    # Partition en train_set et test_set
    return train_test_split(X, y, test_size=0.2, random_state=20)
   