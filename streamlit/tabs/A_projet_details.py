import streamlit as st

from my_functions import insert_head

title = "Le Projet en détails"
sidebar_name = "Le Projet en détails"


def run():

    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        "/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        # XXX

        ## XXX
        """)
