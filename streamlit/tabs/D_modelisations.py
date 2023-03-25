import streamlit as st

from my_functions import insert_head

title = "Modélisations à la demande"
sidebar_name = "Modélisations à la demande"


def run():

    insert_head(
        'https://datascientest.fr/train/assets/logo_datascientest.png',
        "/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        # A REPRENDRE DE PyStreamlit_old.py

        ## XXX
        """)
