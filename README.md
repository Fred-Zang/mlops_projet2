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
1 - aller sur localhost:8000/docs

## C :  voir le streamlit
1 - aller sur localhost:8501


# --------- workflows de github-actions --------- #
## A : Activation du mode debug si pb dans workflow dans le repo git du projet
* (toujours actif normalement car à faire qu'une seule fois pour tous les repo )
sur "mon profil github" : activer notifications  => Settings > notification > Actions > notifications à activer ="automatically watch repositories" + dans system > Actions :Notifications for workflow runs on repositories set up with GitHub Actions => cocher Only notify for failed workflows
* sur le repo du projet : settings > secret and variables > actions > new repository secret
puis  Ajout des 2 configurations suivante dans la fiche : 
a) NAME : ACTIONS_RUNNER_DEBUG  / Value : true  et Clic sur "Add secret"
b) NAME : ACTIONS_STEP_DEBUG  / Value : true  et Clic sur "Add secret"


## B : 1er workflow "1ere action avec tag auto et mise-à-jour fichiers"
1> création du dossier .github/workflows dans rep. principal MLops-Satispy2
2> 1er fichier workflow tag_auto.yml créé dans .github/workflows selon tuto https://github.com/marketplace/actions/tag-release-on-push-action








