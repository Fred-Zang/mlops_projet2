import streamlit as st

title = "Conclusion et Remerciements"
sidebar_name = "Conclusion et Remerciements"


def run():

    st.title(title)

    st.markdown("---")

    st.write("")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.write(
            "Traiter un sujet comme celui-ci a été passionnant et nous aimerions le poursuivre encore")
        st.write("")
        st.write("Nous y avons mis beaucoup de cœur, à découvrir toutes ces notions à travers nos modules de cours et à dérouler une grande quantité de modèles pour trouver des approches et ressources différentes.")
        st.write("")
        st.write("Nous avons le sentiment d’en être qu’au tout début de notre étude, tant il nous reste de questions à résoudre, de tests à réaliser et de nouvelles pistes à explorer.")
        st.write("")
        st.write("Nous espérons que sa lecture vous a été agréable et vous recommandons de parcourir les vidéos et liens donnés en annexe si vous souhaitez en savoir plus sur le sujet.")
        st.write("")
        st.markdown(
            "<h1 style='text-align: center; color: white;'>🌼​ Remerciements 🌼​</h1>",
            unsafe_allow_html=True)
        st.write("")
        st.write("Nous tenons à remercier toute l’équipe de DataScientest pour son écoute et leurs conseils, et particulièrement notre mentor de projet Antoine qui nous a suivi chaque semaine en réunion zoom avec un sourire et une patience admirable.")
