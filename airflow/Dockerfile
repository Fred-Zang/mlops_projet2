FROM apache/airflow:2.5.2
COPY requirements.txt /requirements.txt
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt
RUN python3 -m spacy download fr_core_news_md