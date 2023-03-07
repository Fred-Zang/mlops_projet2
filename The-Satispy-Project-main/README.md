# The Satispy Project

## Explications et Instructions



**Application Streamlit**

Un template d'application [Streamlit](https://streamlit.io/) est disponible dans le dossier [`streamlit_app`](streamlit_app). Vous pouvez partir de ce template pour mettre en avant votre projet.

## Presentation

This repository contains the code for our project **The SatisPy-Project**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to predict a customer's satisfaction. it's a ranking problem :find number of stars of satisfaction of a customer.

This project was developed by the following team :

- Quan Liu ([GitHub](https://github.com/luckychien87) / [LinkedIn](https://www.linkedin.com/in/quan-liu-fr/)))
- Fred Zanghi ([GitHub](https://github.com/Fred-Zang) / [LinkedIn](http://linkedin.com/in/fred-zanghi-89a01390/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

You can also run the Streamlit app in a [Docker](https://www.docker.com/) container. To do so, you will first need to build the Docker image :

```shell
cd streamlit_app
docker build -t streamlit-app .
```

You can then run the container using :

```shell
docker run --name streamlit-app -p 8501:8501 streamlit-app
```

And again, the app should then be available at [localhost:8501](http://localhost:8501).
