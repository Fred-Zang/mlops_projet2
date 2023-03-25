import streamlit as st

from my_functions import insert_head

title = "Cahier des charges"
sidebar_name = "Cahier des charges"


def run():

    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        "/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png")

    st.title(title)

    st.markdown("---")

    st.write("Schéma d'implantation - TOUT A FAIRE !")
    st.image("/airflow/data_others/JPG-PNG/implantation.png")
    st.image("/airflow/data_others/JPG-PNG/Capture d’écran 2023-03-23 183210.png")
    st.write("Suivi du cahier des charges  => à refaire proprement + images")
    st.image("/airflow/data_others/JPG-PNG/cahier_des_charges_1.png")
    st.image("/airflow/data_others/JPG-PNG/cahier_des_charges_2.png")
    st.image("/airflow/data_others/JPG-PNG/cahier_des_charges_3.png")
