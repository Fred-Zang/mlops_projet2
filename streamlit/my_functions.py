import streamlit as st

def insert_head(img1, img2):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(img1)
    with col2:
        st.write(' ')
    with col3:
        st.write(' ')
    with col4:
        st.write(' ')
    with col5:
        st.image(img2)
