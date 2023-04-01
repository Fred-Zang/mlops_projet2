import streamlit as st

title = "Conclusion et Remerciements"
sidebar_name = "Conclusion et Remerciements"


def run():

    st.title(title)

    st.markdown("---")

    st.write(
        """ * :orange[Nous aurions aimé] avoir plus de temps pour optimiser notre code et le transférer sur un Cloud pour simplifier la procédure d'installation.
* La :blue[formation MLOps de DataScientest] est très dense avec de nombreux examens pour valider les modules d'apprentissages.
* Il nous fallait atteindre le dernier module pour avoir enfin une visible d'ensemble et réaliser ce projet en 2 semaines intenses.
* mais ce fût :green[un vrai plaisir] dans le partage, la découverte de nombreux autres concepts pour lever bien des bugs
et la mise en pratique de notre 1er projet MLOps.""")
    st.write("")
    st.write(
        """ * Nous espérons que sa découverte et mise en utilisation vous a été agréable.""")
    st.write("")
    st.markdown(
        "<h1 style='text-align: center; color: white;'>🌼​ Remerciements 🌼​</h1>",
        unsafe_allow_html=True)
    st.write("")
    st.write(
        """ * Nous tenons à :orange[remercier Alban et toute l'équipe de DataScientest] pour cette belle et enrichissante formation.""")
    st.write("")
    st.write("**:green[HappyCoding Yours]** - *Quan , Éric & Fred*")
