# frontend / streamlit
FROM python:3.9

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip install -U pip

# ancienne image OK
RUN pip install -r requirements.txt

# essaie de remplacer nlp = spacy.load('fr_core_news_md') dans le fichier .py
RUN python3 -m spacy download fr_core_news_md

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "PyStreamlit.py"]