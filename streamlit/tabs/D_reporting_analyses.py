import streamlit as st
import pandas as pd

title = "Visual Reporting et Alertes Performances"
sidebar_name = "Visual Reporting et Alertes Performances"


def run():

    st.title(title)

    st.markdown("---")

    path_main = "/airflow/clean_data"
    path_list = [
        "/reporting_GBC2.csv",
        "/reporting_SVM.csv",
        "/reporting_ANN.csv"]
    models_list = ["GBC", "SVM", "ANN"]

    i = 0  # compteur pour la liste des modeles
    for model in models_list:
        # i = 0
        st.write(
            '#### Graphe des mises-à-jour :green[Accuracy - F1-0 - F1-1] du :red[modèle {}]'.format(models_list[i]))

        df = pd.read_csv(path_main + path_list[i])
        st.line_chart(df, x="Date-Heure", y=["f1_0", "f1_1", "accuracy"])

        st.write("")
        # liste des valeurs accuracy, f1-0 et f1-1 de la dernière mise à jour
        exigence_list = list(df.iloc[-1, 1:-1])
        # condition toutes les valeurs de la liste > 0.8
        condition = all(ele > 0.8 for ele in exigence_list)
        # affichage du test d'exigence
        if condition:
            st.write("#### :blue[Parfait !] l':green[accuracy], le :green[f1-score-0] et le :green[f1-score-1] sont :blue[tous > 80%] à la dernière mise-à-jour du modèle. :blue[Critères d'exigence respectés]")
        else:
            st.write(
                "#### :red[ALERTE], au moins une des :red[exigences à 80% n'est pas respectée] sur la dernière mise à jour du modèle")
            st.dataframe(df.iloc[-1, :])
        i = i + 1
        st.markdown("---")
