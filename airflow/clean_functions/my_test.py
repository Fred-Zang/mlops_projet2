##########################################################
#              TESTS UNITAIRES                           #
#                                                        #
# Au lancement de pytest dans le fichier python-app.yml  #
# du répertoire .github\workflows :                      #
# pytest recherche les noms de ficher avec *test.py      #
# et les lance pour tester tous les assert               #
# lors de chaque push ou pull (suivant configuration)    #
#                                                        #
##########################################################


# CHARGEMENT DES PACKAGES
import pandas as pd
import numpy as np

from my_functions import transform_star_to_target


# PRETRAITEMENT DES DONNEES - TRANSFORMATION DES ETOILES EN CIBLE (0 ou 1)
def test_transform_star_to_target():
    """Teste si la transformation d'étoiles en cible est correcte
            1 ou 2 étoiles    => Sentiment négatif (0)
            3, 4 ou 5 étoiles => Sentiment positif (1)
    """
    """ 1 étoile => Cible 0"""
    assert transform_star_to_target(
        pd.DataFrame(
            data=[1],
            columns=['star']))[0] == 0

    """ 2 étoiles => Cible 0"""
    assert transform_star_to_target(
        pd.DataFrame(
            data=[2],
            columns=['star']))[0] == 0

    """ 3 étoiles => Cible 1"""
    assert transform_star_to_target(
        pd.DataFrame(
            data=[3],
            columns=['star']))[0] == 1

    """ 4 étoiles => Cible 1"""
    assert transform_star_to_target(
        pd.DataFrame(
            data=[4],
            columns=['star']))[0] == 1

    """ 5 étoiles => Cible 1"""
    assert transform_star_to_target(
        pd.DataFrame(
            data=[5],
            columns=['star']))[0] == 1

    """ Teste 10 valeurs différentes """
    # DataFrame contenant les valeurs des étoiles
    df_test_star = pd.DataFrame(
        data=[
            1,
            1,
            5,
            4,
            2,
            2,
            3,
            4,
            1,
            2],
        columns=['star'])
    # Liste attendue en sortie
    star_in_target = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    assert not((transform_star_to_target(df_test_star) - star_in_target).any())


# TRAITEMENT DES MOTS VIDES (ou STOP WORDS)
def test_collect_stopwords():
    """Teste si les stopwords renvoyés dans la liste sont bien uniques ???
        Pas trop d'intérêt...
    """
    # Pour tester, besoin de créer un fichier fake de stopwords
    assert True


# REPORTING - PERFORMANCES
def test_reporting():
    """Calcul et affichage du rapport de classification et de la matrice de confusion d'un modèle
        Vérifier qu'aucune erreur n'est générée ???
    """
    assert True


# MODELE PREENTRAINE AVEC WIKIPEDIA - SIMILARITES DE MOTS
def find_similar():
    """Méthode non utilisée, mais possible à tester avec l'exemple de la reine
    """
    assert True


# FONCTION my_doc2vec() DE PRETRAITEMENT
def test_my_doc2vec():
    """Fonction pour transformer un document (ici une ligne de commentaire) en vecteur à partir des tokens qui le compose
    => Faire un test avec doc vide, vecteur null
    => Faire un test avec des valeurs connus
    """
    assert True


# PRETRAITEMENT DU CORPUS ET CREATION DES JEUX D'ENTRAINEMENT ET DE TEST
# POUR LES MODELES SVM ET ANN ###
def prepare_data():
    """Prétraitement du corpus afin de fournir les jeux d'entrainement et de test
    Tester le format des données renvoyées
    """
    assert True
