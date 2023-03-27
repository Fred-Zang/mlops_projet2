import streamlit as st

title = "Améliorations à prévoir et Problèmatiques"
sidebar_name = "Améliorations à prévoir et Problèmatiques"


def run():

    st.title(title)

    st.markdown("---")

    st.write("VULNÉRABILITÉ DES CONTAINERS")
    st.markdown(""" - exemple ici vulnératbilité sur python 3.9""")
    st.image("/airflow/data_others/JPG-PNG/vulnerability_containers.png")

    st.write("POIDS DES IMAGES BUILDÉES")
    st.markdown(""" - Les images contruites pour STreamlit, FastAPI et AirFlow sont assez lourde avec respectivement 1 GB, 2GB et 2GB""")
    st.image("/airflow/data_others/JPG-PNG/Poids_des_images.png")
    st.write('')
    st.write("VERSIONNING DES PACKAGES")
    st.image("/airflow/data_others/JPG-PNG/requirement_streamlit.png")
    st.write('')
    st.write("USAGE DE RAM IMPORTANTE")
    st.markdown(
        """ - problématiques de chargement en local et une utilisation de RAM trop importante si RAM pc < 16 Go """)
    st.image("/airflow/data_others/JPG-PNG/Vmmem.png")
    st.image("/airflow/data_others/JPG-PNG/Vue_processeur.png")
    st.write('')
