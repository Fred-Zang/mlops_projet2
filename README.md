# -----------Présentation rapide du Projet ------------------

- Dans le cadre d'un projet mémoire de fin de formation MLOPS, nous avons récupéré le développement
d'un ancien projet réalisé en formation Data Scientist sur la réalisation de modèles IA d'analyse de 
sentiments.
- Le but n'étant pas d'optimiser ces modèles, mais plutôt de réaliser un ensemble de méthodes de mise 
en production selon le cycle de vie MLOps.
- Nous avons retenu 3 modèles sur 8 testés auparavant : 1 Gradient Boosting Classifier, 1 Support Vector Machine
 et un Artificial Neuronal Network tous 2 lancés après un modèle pré-entrainé wikipedia2vec


# ------------ Lancement de ce REPO Github ------------------ 

## sur un IDE (VSC par exemple) en local
 - $ git clone https://github.com/Fred-Zang/mlops_projet2.git  
    => cela va copier tous les dossier et fichiers du REPO dans un nouveau dossier local "mlops_projet2"

## récupérer wikipedia2vec
- ce modèle pré-entrainé de wikipedia étant assez lourd (+ de 1Go), nous ne pouvons le déposer sur GitHub.
Aussi, veuillez suivre ce lien pour le télécharger en local et placer ce fichier dans le dossier /airflow/clean_model
Ceci est obligatoire et doit être réalisé des la copie du repo en local sur votre machine.
- LIEN A DONNER ...............!!!!!

# --- voir les navigateurs FastApi et Streamlit tourner ------ 
## A : Création des containers FastApi et Streamlit
 - $ cd mlops_projet2  # se placer sur le dossier de travail du repo
 - $ docker network create AIservice  # créer le reseau AIservice
 - $ docker network ls   # vérification
 - $ docker-compose up -d --build  # création des images et lancement des containers fastapi et streamlit
    => cela peut prendre entr 5 à 10 minutes la 1ere fois selon le PC

ATTENTION : si votre PC n'a pas plus de 16Go de RAM, cela peut être délicat pour la suite.
aussi nous mettons à votre disposition les images déjà construites et informations nécessaire en partie D. REMARQUES

## B : voir la console FastAPI sur un navigateur
- aller sur http://localhost:8000/docs

## C :  voir la console Streamlit sur un navigateur
- aller sur http://localhost:8501

## D : Création des containers Airflow ( 7 containers ici)
- $ cd airflow   # se placer sur le sous-dossier /airflow
- $ docker-compose up -d --build  # création des images et lancement des containers airflow

REMARQUES : Nous n'avons pas utilisé d'images alpine dès le début dans notre projet, d'où un poids important des images crées
et une utilisation de ressources RAM importante. Désolé. Devant rendre le projet rapidement maintenant, nous vous proposons de
récupérer les images contruiste via DockerHub (créez un compte si besoin).

- il vous faudra alors modifier les codes des docker-compose en remplaçant 
build : /fastapi    par    image: fredzang/my_fastapi_mlops:2.0.0     # dans le docker-compose principal
build : /streamlit  par    image: fredzang/my_streamlit_mlops:2.0.0   # dans le docker-compose principal
build: .   par   image: ${AIRFLOW_IMAGE_NAME:my_airflow_mlops:2.0.0}  # dans le docker-compose de airflow

## E : voir la console Airflow sur un navigateur
- aller sur http://localhost:8080

## F : par la suite
- une fois toutes ces étapes réalisées, ce sera bien plus simple par la suite :
- il vous suffira de lancer les 2 docker-compose pour relancer tous les containers et accèder aux consoles sur votre navigateur,
avec un bien meilleur délai de constructions

# --------------- Arborescence et Fonctionnement ------------

## Gestion de nos bases de Données
- N'ayant pas accès à l'API des données, nous avons reçu une base de 17200 lignes que nous avons découpé en 3 parties
pour simuler les mises-à-jours de notre base de données, des modélisations et reporting selon des métrics et exigences
de performances que nous avons définis.

- Ainsi, Fastapi gère notre base de données, sa mise-à-jour par ajout d'un fichier csv de nouvelles données
et la possibilité d'ajouter un commentaire et sa notation directement à data_MAJ.csv qui est la base de référence
actualisée et utilisé par Streamlit et Airflow par la suite.

- Streamlit nous sert de présentation détaillée du projet, des problématiques rencontrées et améliorations à prévoir
ainsi que de prédiction à la demande sur les 3 modèles retenus et analyse du reporting des performances de chacun.

- Airflow nous sert de pipeline CI/CD afin de réaliser les preporcessings, les training et reporting de nos 3 modèles.
Les modèles mis-à-jour sont sauvegardés avec date et heure de réalisation dans /airflow/clean_model
Les reporting mis-à-jour sont sauvegardés avec date et heure de réalisation dans /airlow/clean_data

## -------------- Nous aurions aimé faire bien plus ---------
- Nous nous excusons par avance de ne pas avoir eu le temps d'optimiser au mieux notre code.
La formation MLOps de DataScientest est très dense avec de nombreux examens pour valider les modules d'apprentissages.
Il nous fallait atteindre le dernier module pour avoir enfin une visible d'ensemble et réaliser ce projet en 2 semaines
intenses, mais ce fût un vrai plaisir dans le partage, la découverte de nombreux autres concepts pour lever bien des bugs
et la mise en pratique de notre 1er projet MLOps.
- Merci à Alban et toute l'équipe de DataScientest pour cette belle formation.

Quan, Éric & Fred la Satispy Team ;-)
