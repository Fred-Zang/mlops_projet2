# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 01:41:01 2022

@author: user
"""
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi import UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import csv
import datetime

app = FastAPI(
    title="Backend FastAPI by Eric & Fred & Quan",
    description="API for MLOps final project: SatisPy",
    version="1.0.0",
    openapi_tags=[
        {
            'name': "Authentification"
        },
        {
            'name': 'Data Management',
            'description': "Data management for sentiment analysis"
        }
    ]
)

security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# dictionnaire des users avec hashage du mot de pass
# deux utilisateurs: user normale ("user") et administateur ("admin")

users = {
    "user": {
        "username": "user",
        "hashed_password": pwd_context.hash("user")
    },
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash('admin')

    }
}


# ----------------------------------1ere route Authentification ------------------------------------------------ #

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """_summary_ On récupère dans la variable credentials les identifiants entrés par l'utilisateur à l'aide de la dépendance \
Depends(security). On exige par cette dépendance, l'authentification de l'utilisateur à l'aide de la méthode HTTP. \
On récupère l'identifiant et le mot de passe de l'utilisateur grâce aux attributs username et password de la variable \
credentials. \
On vérifie si l'identifiant est présent dans la base de données. Ensuite, on compare si le mot de passe crypté correspond bien \
à celui de la base de données en utilisant la méthode verify de la variable pwd_context.  \
Sinon, on renvoie l'identifiant de l'utilisateur.

    Args:
        credentials (HTTPBasicCredentials, optional): _description_. Defaults to Depends(security).

    Raises:
        HTTPException: On lève une erreur 401 si username et hashedpassword ne correspondent pas

    Returns:
        credentials.username
    """

    username = credentials.username
    if not (
        users.get(username)) or not (
        pwd_context.verify(
            credentials.password,
            users[username]['hashed_password'])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get('/status', tags=['Authentification'])
async def get_status(username: str = Depends(get_current_user)):
    """
    _summary_ : pour vérifier l'état d'API

        API status

    Args:\n
    1. username (str, optional): _description_ - Defaults to Depends(get_current_user).

    Returns: \n
    _type_: string  _description_ - "API is ready" for user or admin OR "API is not ready for unknown user"
    """
    if username == "admin" or username == "user":
        return "API is ready"
    else:
        return "API is not ready for unknown user"


@app.get("/authentification", name="Hello", tags=['Authentification'])
async def current_user(username: str = Depends(get_current_user)):
    """
    _summary_ : pour accéder à cette route, il faut au préalable que l'utilisateur se soit authentifié.

        HELLO user

    Args:\n
    username (str, optional): _description_ - Defaults to Depends(get_current_user).

    Returns:\n
    _type_: string  _description_- affiche "Hello" + "username"
    """
    return "Hello {}".format(username)


# ----------------------------------2eme route Création de la base de donnée ------------------------------------------------ #
dt = datetime.datetime.now()
time = dt.strftime('%d/%m/%Y')
# # data used in GBC, ANN and SVM


class Item(BaseModel):
    inID: Optional[int] = None
    Commentaire: str = Field(..., min_length=1)  # at least one world
    star: int = Field(None, ge=1, le=5)
    source: Optional[str] = None
    company: Optional[str] = None
    Index_org: Optional[int] = None
    Star_org: Optional[int] = None
    date: str = Field(
        default=time,
        regex="^\\d{2}\\/\\d{2}\\/\\d{4}$")  # the default is today

# # data base  information
# ################################################################################
# # modify this adresse to airflow
# data_store_path = '/airflow/clean_data/'
# # modify this file to origin database
# data_name = 'data_MAJ.csv'
# ################################################################################


def write_comment(FileName: str, inputs: list):
    with open(FileName, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            inputs.inID,
            inputs.Commentaire,
            inputs.star,
            inputs.source,
            inputs.company,
            inputs.Index_org,
            inputs.star,  # the star_org is a copy of star for furture verification
            inputs.date])


@app.post('/comment', name="Make a New comment", tags=['Data Management'])
async def post_comment(item: Item, username: str = Depends(get_current_user)):
    """
    _summary_ : pour déposer un nouveau commentaire par user or admin

        BASE DE DONNEE mise à jour

    Args:
    1. username (str, optional): _description_  Defaults to Depends(get_current_user).\n
    2. item (BaseModel): _description_ Commentaire modèle\n
        _Détail sur item_ :\n
        inID: Optional[int] - inID sera màj automatiquement \n
        Commentaire: str - commentaire du client\n
        star : int - étoile donnée par le client \n
        source: Optional[str] - source de commentaire (ex:TRUSTPILOT ou TRUSTED SHOPS ou None)\n
        company: Optional[str] - site web (ShowRoom ou VeePee ou None) \n
        Index_org: Optional[int] - ID origine \n
        Star_org: Optional[int] - étoile origine avant transférer en 0 et 1, un cope de "star"\n
        date: str - date de commentaire, dd/mm/yyyy\n

    Returns: \n
    _type_: dict - {"nouveau commentaire ID" : int, "nouveau commantaire": str} si authentifié
    """
    # data storage file name
    ##########################################################################
    # modify this adresse to airflow
    data_store_path = '../airflow/clean_data/'
    # modify this file to origin database
    data_name = 'data_MAJ'
    data_type = '.csv'
    ##########################################################################

    dt = datetime.datetime.now()
    time = dt.strftime('%Y-%m-%d-%H-%M')

    data_file = data_store_path + data_name + data_type
    data_file_time = data_store_path + data_name + '_' + time + data_type

    # print(data_file_time)
    if username == 'admin' or username == "user":
        write_comment(data_file, item)
        # modify the InID
        comments = pd.read_csv(data_file)
        comments['inID'] = range(len(comments))

        comments.to_csv(data_file, index=False)
        comment = comments.iloc[-1, :]

        # database backup with current time
        comments.to_csv(data_file_time, index=False)

        return {
            'last comment ID': int(
                comment['inID']),
            'last comment': comment['Commentaire']}
    else:
        return "Sorry you have no rights!"


@app.post("/uploadfile", name='upload file', tags=['Data Management'])
async def create_upload_file(file: UploadFile, username: str = Depends(get_current_user)):
    """
    _summary_ : pour déposer des nouveaux commentaires via un csv document par un admin

        BASE DE DONNEE data_MAJ.csv mis à jour par csv
        + création archive data_MAJ_annee-moi-jour-heure:min.csv

    Args:\n
    1. username (str, optional): _description_ - Defaults to Depends(get_current_user).\n
    2. file (csv): _description_ - csv document avec plusieurs commentaires\n

    Returns: \n
    si authentifié et un csv => un filename pour indiquer OK

    PS: le fichier téléchargé est sauvegardé avec l'heure actuelle
    """

    ##########################################################################
    # modify this adresse to airflow
    data_store_path = '../airflow/clean_data/'
    # modify this file to origin database
    data_name = 'data_MAJ'
    data_type = '.csv'
    ##########################################################################

    print(file.content_type)
    if username == 'admin':
        data_file = data_store_path + data_name + data_type
        dt = datetime.datetime.now()
        # if the hour and minute cause traitement problem, we can keep only
        # y-m-d
        time = dt.strftime('%Y-%m-%d-%H-%M')
        data_file_time = data_store_path + data_name + '_' + time + data_type
        file.filename = 'new_data_' + time + data_type

        if file.content_type == "text/csv":
            # write into a new data file
            with open(data_store_path + file.filename, 'wb') as f:
                f.write(await file.read())

            # update the data_MAJ file (global data base)
            added_data = pd.read_csv(data_store_path + file.filename)
            added_data.to_csv(data_file, mode='a', header=False, index=False)

            # modify the ID
            data = pd.read_csv(data_file)
            data['inID'] = range(len(data))
            data.to_csv(data_file, index=False)
            data.to_csv(data_file_time, index=False)

            return {"filename": file.filename}
        else:
            # elif file.content_type =='application/json':
            return {'INFO': 'please use a csv file'}
    elif username == 'user':
        return "Only admin can upload a file, please contact your admin or upload one note once!"

'''
@app.get('/getcomments', name= 'Get comments', tags = ['Data Management'])
def get_comments(username: str = Depends(get_current_user)):
    """
    _summary_ : sauvegarde en json de data_MAJ.csv si authentifié

        data_MAJ.csv => data_MAJ.json

    Args: \n
    1. username (str, optional): _description_ - Defaults to Depends(get_current_user).

    Returns: \n
    _type_: json - un json si authentifié
    """
    # data storage file name
    ################################################################################
    # modify this adresse to airflow
    data_store_path = '../airflow/clean_data/'
    # modify this file to origin database
    data_name = 'data_MAJ.csv'
    data_file_csv = data_store_path+data_name
    # modify the json name
    data_file_json = data_store_path + 'data_MAJ.json'
    ################################################################################

    if username == 'admin' or username =='user':
        comments = pd.read_csv(data_file_csv)

        data_json = comments.to_json(data_file_json)

        data = comments.to_json()

        return data
    else:
        return "Sorry you have no rights!"
'''
