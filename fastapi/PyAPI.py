# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 01:41:01 2022
@author: user
"""
from fastapi import Depends, FastAPI, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
import pandas as pd 
import numpy as np
import csv 
import json

app = FastAPI(
    title="Backend FastAPI by Eric & Fred & Quan",
    description = "API for MLOps final project: SatisPy",
    # version="wikihappy.org"
    version = "1.0.0",
    openapi_tags=[
        {
            'name': "Authentification"
        },
        {
            'name': 'Data Management',
            'description':"Data management for sentiment analysis"
        }
    ]
)

security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# dictionnaire des users avec hashage du mot de pass
# deux utilisateurs: user normale ("user") et administateur ("admin")

users = {
    "user" : {
        "username" : "user",
        "hashed_password" : pwd_context.hash("user")
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
On récupère l'identifiant et le mot de passe de l'utilisateur grâce aux attributs username et password de la variable credentials \
On vérifie si l'identifiant est présent dans la base de données. Ensuite, on compare si le mot de passe crypté correspond bien à \
celui de la base de données en utilisant la méthode verify de la variable pwd_context.  \
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

@app.get('/status',tags = ['Authentification'])
async def get_status(username: str = Depends(get_current_user)):
    
    if username== "admin" or username=="user":
        return "API is ready"
    else: 
        return "API is not ready for unknown user"

@app.get("/authentification", name="Hello", tags=['Authentification'])
async def current_user(username: str = Depends(get_current_user)):
    """
    _summary_ : pour accéder à cette route, il faut au préalable que l'utilisateur se soit authentifié.
        HELLO user
    Args: username (str, optional): _description_. Defaults to Depends(get_current_user).
    Returns: _type_: string  _description_: affiche "Hello" + "username"
    """
    return "Hello {}".format(username)


# ----------------------------------2eme route Création de la base de donnée ------------------------------------------------ #


# @app.post("/Postez un commentaire",
#           name="Nouveau Commentaire ",
#           tags=['admin'])
# async def admin_user(comment: str, username: str = Depends(get_current_user)):
#     """
#     _summary_ : détails à donner


#     Args: idem
#     """

#     # ? ajouter la date de saisie du commentaire en return
#     return {username: comment}

# # original data set
# #Commentaire	star	date	client	reponse	source	company	ville	maj	date_commande	ecart
# class Item(BaseModel):
#     # inID: Optional[int] = None 
#     Commentaire: str
#     star : int
#     date: Optional[str] = None
#     client: Optional[str]= None
#     reponse: Optional[str]= None
#     source: Optional[str] = None
#     company: Optional[str] = None
#     ville: Optional[str]=None
#     maj: Optional[str]=None
#     date_commande: Optional[str] = None 
#     ecart: Optional[int]=None

# # data used in GBC, ANN and SVM 
class Item(BaseModel):
    inID: Optional[int] = None 
    Commentaire: str
    star : int
    source: Optional[str] = None
    company: Optional[str] = None
    Index_org: Optional[int] = None
    Star_org: Optional[int] =None
    date: Optional[str] = None

# data base  
data_store = '/airflow/clean_data/data_MAJ.csv'

# function definition
# function to store a new comment
# def write_comment(FileName : str, inputs:list):
#     with open(FileName,'a+') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([inputs.Commentaire, 
#         inputs.star, 
#         inputs.date, 
#         inputs.client, 
#         inputs.reponse, 
#         inputs.source, 
#         inputs.company,
#         inputs.ville,
#         inputs.maj,
#         inputs.date_commande,
#         inputs.ecart])

def write_comment(FileName : str, inputs:list):
    with open(FileName,'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
        inputs.inID,
        inputs.Commentaire, 
        inputs.star,           
        inputs.source, 
        inputs.company,
        inputs.Index_org,
        inputs.Star_org,
        inputs.date])




@app.post('/comment', name="Make a New comment", tags= ['Data Management'])
async def post_comment(item:Item):
    # data storage file name
    data_store = '/airflow/clean_data/data_MAJ.csv' # change name data_MAJ.csv

    write_comment(data_store,item)
    comments = pd.read_csv(data_store)
    comment = comments.iloc[-1,:]
    return {'new comment': comment['Commentaire']}

@app.get('/getcomments', name= 'Get comments', tags = ['Data Management'])
def get_comments():
    # data storage file name
    data_store = '/airflow/clean_data/data_MAJ.csv'
    comments = pd.read_csv(data_store)
    # comment = comments.iloc[-1,:]
    # total = len(comments.Commentaire)
    data_json = comments.to_json('/airflow/clean_data/data_MAJ.json')

    data = comments.to_json()
    
    return data