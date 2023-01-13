# A : REPRISE DE MES DOSSIERS EN LOCAL POUR CREER UN NOUVEAU REPO GITHUB

1 – création d’un REPO github vide nommé mlops_projet2 sur https://github.com/Fred-Zang/mlops_projet2

2 – création en local du dossier mlops_projet2 avec dedans tous les dossiers et fichiers souhaités

3 – EN LOCAL sur   fred07@PCwikHap:~/DataScientest_fred07/mlops_projet2$

$ git init  # pour créer un .git à la racine du dossier de travail et pouvoir relier ce dossier au repo git
->	Initialized empty Git repository in /home/fred07/DataScientest_fred07/mlops_projet2/.git/

$ git status # => on voit que tous les dossiers et fichiers sont Untracked files + No commit yet

$ git add –all  # pour tout tracker

$ git commit -am "1er_commit_tous_dossier&fichier_local"  # pour un 1er snapshot de tous les add

$ git remote add remote1 https://github.com/Fred-Zang/mlops_projet2.git   # prépare l’envoie distant nommé remote1

$ git push -u remote1 master  # envoyer le remote1 sur mon repo github vierge => tout mon local se copie sur github
->	On voit sur mon github perso tous les dossier et fichier du PC local qui sont copiés

# B : mettre à jour le repo github 

1 - modification ou création de fichiers (dossiers) sur PC local

2 - si nouveau fichier jamais tracké

$ git status # pour voir les modifs à pusher $ git add <chemin/fichier.extension> ou $ git add --all si c'est ok pour tout prendre $ git commit -m "explication_du_commit" # pour un xieme snapshot $ git remote add remoteX https://github.com/Fred-Zang/mlops_projet2.git # si fichier jamais tracké ! $ git push -u remoteX master

2- bis si fichier déjà tracké et pushés

$ git commit -am "commentaire" $ git push

C : ---------------------- mettre à jour le repo local --------------------------
1 – ajout de fichier dans le repo github

2 - sur la terminal en local

$ git fetch # pour vérifier les modifs sur le repo github $ git pull # fait la mise à jour sur le repo local

$ git reset HEAD~1 = effacer un commit non pushé