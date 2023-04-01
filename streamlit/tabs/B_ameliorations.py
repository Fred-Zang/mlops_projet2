import streamlit as st

title = "Améliorations à prévoir et Problèmatiques"
sidebar_name = "Améliorations à prévoir et Problèmatiques"


def run():

    st.title(title)

    st.markdown("---")

    st.markdown("❌ :red[1 -PROBLÈME DE POIDS DES IMAGES BUILDÉES]")
    st.markdown(
        """ - Les images contruites pour Streamlit, FastAPI et AirFlow font :orange[plus de 18 GB] utilisées.""")
    st.markdown(
        """ - On remarque que Airflow requiert la grande majorité de cette mémoire avec 7 containers.""")
    st.image("/airflow/data_others/JPG-PNG/bilan-images-poids-x9.png")

    st.write('')
    st.write("❌ :red[2 -PROBLÈME D'USAGE DE RAM IMPORTANTE]")
    st.markdown(
        """ - Le :orange[processus Vmmem] lancé par Docker nous demande une utilisation de RAM trop importante si la RAM PC < 32 Go """)
    st.markdown(
        """ - Détails ci-dessous sur un PC disposant de :green[32Go de RAM] du processus Vemmem et répartition par containers au démarrage + en charge de travail """)

    st.image("/airflow/data_others/JPG-PNG/Vmmem.png")
    st.image("/airflow/data_others/JPG-PNG/bilan-memory-8-containers.png")

    st.write('')
    st.write("❌ :red[3 -PROBLÈME DE POIDS DE LA DISTRIBUTION DOCKER-DESKTOP]")
    st.markdown(
        """ - Utiliser Docker-Desktop est très pratique pour la gestion de nos containers mais cela nécessite une bonne capacité de Disque Dur, avec :orange[90 Go] utilisé sur 1 de nos PC""")
    st.image("/airflow/data_others/JPG-PNG/probleme_poids_DD_docker_desktop.png")

    st.write('')
    st.write("✔️ :green[AMÉLIORATIONS ENVISAGÉES A CE STADE :]")
    st.markdown(
        """ - Basculer Airflow sur un :blue[Cloud ou un serveur distant]""")
    st.markdown(""" - Déplacez la distribution docker-desktop-data hors du lecteur système (voir https://dev.to/kimcuonthenet/move-docker-desktop-data-distro-out-of-system-drive-4cg2) """)

    st.write('')
    st.write("❌ :red[4 -PROBLÈME DE VERSIONNING DES PACKAGES UTILISÉS]")
    st.markdown(
        """ - le poids des images docker et l'utilisation importante de RAM est aussi dûe à nos :orange[versions de packages] utilisés""")
    st.markdown(
        """ - Nous avons commencé à faire des tests pour descendre les versions sklearn, tensorflow ect... mais cela :orange[nécessite encore bien des tests]""")
    st.image("/airflow/data_others/JPG-PNG/requirements_files.png")

    st.write('')
    st.write(
        "✔️ :green[AMÉLIORATIONS ENVISAGÉES POUR LES VERSIONS DE PACKAGES :]")
    st.markdown(
        """ - construire 1 ou des :blue[containers dédiés aux packages] de preprocessings et modélisations""")
    st.markdown(
        """ - nous pourrions par exemple tester ce container dédié à tensorflow https://hub.docker.com/r/tensorflow/build""")

    st.write('')
    st.write("❌ :red[5 -VULNÉRABILITÉS DES IMAGES DOCKER]")
    st.markdown(
        """ - Docker-desktop nous permet d'analyser les vulnérabilités de nos images déployées""")
    st.markdown(
        """ - Nous trouvons : 24 Light pour :blue[FastAPI], :orange[7 Critic] + 77 High + 27 Medium et 119 Light pour :blue[Streamlit] par exemple ...""")
    st.image("/airflow/data_others/JPG-PNG/vulnerability_containers_streamlit.png")

    st.write('')
    st.write("✔️ :green[AMÉLIORATIONS ENVISAGÉES POUR LES VULNÉRABILITÉS :]")
    st.markdown(
        """ - Il existe des outils de :blue[scan des images] que nous ne connaissions pas et qu'il nous faut apprendre à manipuler""")
    st.markdown(""" - et nous devons étudier la documentation Docker avant tout : https://docs.docker.com/engine/scan/#:~:text=Docker%20Scan%20versions%20earlier%20than,Docker%20Desktop%20to%20version%204.3.""")

    st.write('')
    st.write("❌ :red[6 -DÉCOUPER NOS FONCTIONS POUR PYTEST]")
    st.markdown(
        """ - Nous n'avons :orange[pas pensé] à découper suffisamment nos fonctions Python pour les rendre testables avec Pytest""")
    st.markdown(""" - Nous nous en sommes aperçu un peu tardivement, désolé pour ce point de négligence qui nous à permis d'apprendre malgré tout""")
