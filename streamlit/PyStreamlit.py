from collections import OrderedDict

import streamlit as st
import config

from tabs import A_projet_details, B_cahier_charges, C_ameliorations, D_modelisations, E_conclusion


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

TABS = OrderedDict(
    [
        (A_projet_details.sidebar_name, A_projet_details),
        (B_cahier_charges.sidebar_name, B_cahier_charges),
        (C_ameliorations.sidebar_name, C_ameliorations),
        (D_modelisations.sidebar_name, D_modelisations),
		(E_conclusion.sidebar_name, E_conclusion)
    ]
)


def run():
    st.sidebar.image(
        "/airflow/data_others/JPG-PNG/logo_SatisPy_Project.png",
        width=150,
    )
    st.sidebar.markdown(
        """
        # Le projet SatisPY
        """)
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
