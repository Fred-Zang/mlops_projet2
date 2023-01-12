# ------------ aide de départ ----------------------- #

# A - si besoin de copier la totalité de ce REPO sur ton github perso
## sur un navigateur
 - aller sur ce github https://github.com/Fred-Zang/MLops-Satispy2
 - s'indentifier 
 - cliquer sur "fork" ( en haut-droite ) du github 
    => cela clone la totalité de son REPO sur ton github perso

# B - copier ce REPO sur ta machine en local
## sur un IDE VSC en local
 - se placer à la racine de ton dossier de travail ( pas besoin de créer un autre dossier )
 - $ git clone https://github.com/Fred-Zang/MLops-Satispy2.git  
    => cela va copier tous les dossier et fichiers du REPO dans un nouveau dossier local "MLops-Satispy2"

## Création des images FastApi et Streamlit par docker-compose
 - $ cd MLops-Satifpy2
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

# voir l'API
1 - aller sur localhost:8000/docs

# voir le streamlit
1 - aller sur localhost:8501

# A faire si possible rapidement :
    - voir comment tagger tout nouvelle image dès qu'on faut 1 modif de fichiers 
    - voir comment synchroniser toutes modifs entre PC local et repo github
    - voir dés le début à effectuer les tests unitaires, d'intégration etc ...
    = créer une action qui exécute des tests avec pytest et contrôle la qualité de votre code avec Flake8 lorsque vous chargez sur votre dépôt.
    => voir github action
    
# Urgent à voir
? utiliser Jenkins, Kubernetes et Airflow => ne vont-ils pas modifier l'architecture de notre github, de nos docker-compose,de nos fichier et dossier ?
ce serait bon de le savoir pour éviter de refaire bien des choses ensuite.
Je vais demander à Alban et/ou Daniel


# 11/01/2023 - Activation du mode debug dans le repo git du projet
sur le repo du projet : settings > secret and variables > actions > new repository secret
puis  Ajout des 2 configurations suivante dans la fiche : 
a) NAME : ACTIONS_RUNNER_DEBUG  / Value : true  et Clic sur "Add secret"
b) NAME : ACTIONS_STEP_DEBUG  / Value : true  et Clic sur "Add secret"


# 12/01/2023 - 1er workflow "1ere action avec tag auto et mise-à-jour fichiers"
1> création du dossier .github/workflows dans rep. principal MLops-Satispy2
2> 1er fichier workflow tag_auto.yml créé dans .github/workflows selon tuto https://github.com/marketplace/actions/tag-release-on-push-action
3> mofication du dossier data ( création dedans du dossier Modèles) + ajout ANN.pickle pour tester
4> $ git commit  -am "1ere action avec tag auto et mise-à-jour fichiers"
   $ git push
=> erreur failed to push some refs to 'https://github.com/Fred-Zang/MLops-Satispy2.git': Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .github/
        data/

5> je fais donc un add de ces dossiers comme demandé :









