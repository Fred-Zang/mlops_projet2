import streamlit as st

title = "Améliorations à prévoir et Problèmatiques"
sidebar_name = "Améliorations à prévoir et Problèmatiques"


def run():

    st.markdown(
        "<h1 style='text-align: center; color: white;'>Améliorations à prévoir et Problèmatiques</h1>",
        unsafe_allow_html=True)

    st.markdown("❌ :red[PROBLÈME DE POIDS DES IMAGES BUILDÉES]")
    st.markdown(""" - Les images contruites pour STreamlit, FastAPI et AirFlow avec plus de 18 GB utilisées.""")
    st.markdown(""" - On remarque que Airflow requière la grande majorité de cette mémoire.""")
    st.image("/airflow/data_others/JPG-PNG/bilan-images-poids-x9.png")

    st.write('')
    st.write("❌ :red[PROBLÈME D'USAGE DE RAM IMPORTANTE]")
    st.markdown(""" - Le processus "Vmmem" lancé par Docker nous demande une utilisation de RAM trop importante si la RAM PC <= 16 Go """)
    st.markdown(""" - Détails ci-dessous sur un PC disposant de 32Go de RAM du porecessus Vemmem et répartition par containers au démarrage et en charge de travail """)
    st.image("/airflow/data_others/JPG-PNG/Vmmem.png")
    st.image("/airflow/data_others/JPG-PNG/bilan-memory-8-containers.png")

    st.write('')
    st.write("❌ :red[PROBLÈME DE POIDS DE LA DISTRIBUTION DOCKER-DESKTOP]")
    st.markdown(""" - Utiliser Docker-Desktop est très pratique pour la gestion de nos containers mais cela nécessite une bonne capacité de Disque Dur""")
    st.image("/airflow/data_others/JPG-PNG/probleme_poids_DD_docker_desktop.png")
    
    st.write('')
    st.write("✔️ :green[AMÉLIORATIONS ENVISAGÉES A CE STADE :]")
    st.markdown(""" - Basculer Airflow sur un Cloud ou un serveur distant""")
    st.markdown(""" - Déplacez la distribution docker-desktop-data hors du lecteur système (voir https://dev.to/kimcuonthenet/move-docker-desktop-data-distro-out-of-system-drive-4cg2) """)

    st.write('')
    st.write("❌ :red[PROBLÈME DE VERSIONNING DES PACKAGES UTILISÉS]")
    st.markdown(""" - le poids des images docker et l'utilisation important de RAM est aussi dû à nos versions de packages utilisés""")
    st.markdown(""" - Nous avons commencé à faire des tests pour descendre les versions sklearn, tensorflow ect... mais cela nécessite encore bien des tests""")
    st.image("/airflow/data_others/JPG-PNG/requirements_files.png")
    
    st.write('')
    st.write("✔️ :green[AMÉLIORATIONS ENVISAGÉES POUR LES VERSIONS DE PACKAGES :]")
    st.markdown(""" - construire 1 ou des containers dédiés aux parckages de preprocessings et modélisations""")
    st.markdown(""" - nous pourrions par exemple tester ce contenaire dédié à tendorflow https://hub.docker.com/r/tensorflow/build""")   
    
    st.write('')
    st.write("❌ :red[VULNÉRABILITÉS DES IMAGES DOCKER]")
    st.markdown(""" - Docker-desktop nous permet d'analyser les vulnérabilités de nos images déployées""")
    st.markdown(""" - Nous trouvons : 24 Light pour FastAPI, 7 Critic + 77 High + 27 Medium et 119 Light pour Streamlit par exemple ...""")
    st.image("/airflow/data_others/JPG-PNG/vulnerability_containers_streamlit.png")
 
    st.write('')
    st.write("✔️ :green[AMÉLIORATIONS ENVISAGÉES POUR LES VULNÉRABILITÉS :]")
    st.markdown(""" - Il existe des outils de scan des images que nous ne connaissions pas et qu'il nous faut apprendre à manipuler""")
    st.markdown(""" - et nous devons étudier la documentation Docker avant tout : https://docs.docker.com/engine/scan/#:~:text=Docker%20Scan%20versions%20earlier%20than,Docker%20Desktop%20to%20version%204.3.""")    
    