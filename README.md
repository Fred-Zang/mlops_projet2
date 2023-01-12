# ------------ Récupérer le REPO Github --------------------- #

## A : sur un navigateur
 - aller sur ce github https://github.com/Fred-Zang/mlops_projet2
 - s'indentifier 
 - cliquer sur "fork" ( en haut-droite ) du github 
    => cela clone la totalité de son REPO sur ton github perso

## B : sur un IDE VSC en local, copier ce REPO sur ta machine en local
 - se placer à la racine de ton dossier de travail ( pas besoin de créer un autre dossier car le dossier mlops_projet2 va se créer tout seul en local )
 - $ git clone https://github.com/Fred-Zang/mlops_projet2.git  
    => cela va copier tous les dossier et fichiers du REPO dans un nouveau dossier local "MLops-Satispy2"

# --- voir les navigateur FastApi et Streamlit tourner ------ #
## A : Création des images FastApi et Streamlit par docker-compose
 - $ cd mlops_projet2
 - créer tout d'abord le reseau AIservice avant de lancer le docker-compose    
 - $ docker network create AIservice
 - $ docker network ls   ( pour vérifier )
 - lancer le docker-compose pour créer toutes les images ( frontend et backend etc ...)
    $ docker-compose up -d --build
    => parfait en 4-5 minutes
    => sauf petit conseil en court de route :
    """ WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. 
    It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
    WARNING: You are using pip version 22.0.4; however, version 22.3.1 is available.
    You should consider upgrading via the '/usr/local/bin/python -m pip install --upgrade pip' command. """
    => donc mise à jour de pip recommandée

## B : voir l'API
1 - aller sur http://localhost:8000/docs

## C :  voir le streamlit
1 - aller sur http://localhost:8501


# --------- workflows de github-actions --------- #
## A : Activation du mode debug si pb dans workflow dans le repo git du projet
* (toujours actif normalement car à faire qu'une seule fois pour tous les repo )
sur "mon profil github" : activer notifications  => Settings > notification > Actions > notifications à activer ="automatically watch repositories" + dans system > Actions :Notifications for workflow runs on repositories set up with GitHub Actions => cocher Only notify for failed workflows
* sur le repo du projet : settings > secret and variables > actions > new repository secret
puis  Ajout des 2 configurations suivante dans la fiche : 
a) NAME : ACTIONS_RUNNER_DEBUG  / Value : true  et Clic sur "Add secret"
b) NAME : ACTIONS_STEP_DEBUG  / Value : true  et Clic sur "Add secret"


## B : 1er workflow "1ere action avec tag auto et mise-à-jour fichiers"
1 - création du dossier .github/workflows dans le dossier de travail mlops_projet2
2 -	Création fichier python-app.yml  # 1er workflow
- sur https://github.com/Fred-ZAng/mlps_projet2
- cliquer Actions > new workflow > "choisir python application" > configure > start commit
-	=> installation flake8 et pytest
-	=> python-version: "3.10"  => voir si pb 3.9 ou 3.9 selon modèles
3	Création fichier_test.py
-	def total() et def test_total() avec assert de PyTest # fichier de test .py

.... expliquer ce fichier yaml + lancement dès le commit + fichier logs => run OK mais beaucoup d'erreurs 
d'indentation, espace etc; dans les fichiers python


# -----------  autopep8 de tous les fichiers python -------------
pour corriger les erreur vu dans les logs de test_python
1 - sur /mlops_projet2$ 
$ autopep8 --in-place -a -a streamlit/PyStreamlit.py
$ autopep8 --in-place -a -a fastapi/PyAPI.py
$ autopep8 --in-place -a -a data/fichier_test.py
=> les 3 fichiers python sont maintenant aux normes pep8

2 - commit du local sur repo github pour vérif des logs






