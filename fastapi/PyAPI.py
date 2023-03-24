# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 01:41:01 2022

@author: user
"""
from fastapi import Depends, FastAPI, HTTPException status, Query
from fastapi import File, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
import pandas as pd 
import numpy as np
import csv 
import json
import datetime

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

# data base  information
########################################
# modify this adresse to airflow
data_store_path = '../airflow/clean_data/'
# modify this file to origin database 
# => data_MAJ.csv
# this is the test small data base
data_name = 'data_MAJ2.csv'
########################################  


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
    ########################################
    # modify this adresse to airflow
    data_store_path = '../airflow/clean_data/'
    # modify this file to origin database
    # => data_MAJ.csv
    # this is a small test data base
    data_name = 'data_MAJ2.csv'
    ########################################  
    
    
    data_file = data_store_path+data_name
    print(data_file)
    write_comment(data_file,item)
    # modify the InID 
    comments = pd.read_csv(data_file)
    comments['inID']= range(len(comments))
    # comments.iloc[-1,0]=comments.iloc[-2,0]+1
    comments.to_csv(data_file,index=False) 
    comment = comments.iloc[-1,:]
    
    return {'last comment ID': int(comment['inID']),'last comment': comment['Commentaire']}
    


@app.get('/getcomments', name= 'Get comments', tags = ['Data Management'])
def get_comments():
    """"
    return comment data base by json & create data_MAJ2.json on local VSC

    """
    # data storage file name
    ########################################
    # modify this adresse to airflow
    data_store_path = '../airflow/clean_data/'  # ce chemin =>  en local sur VSC
    # modify this file to origin database
    # => data_MAJ.csv
    # this is a small database
    data_name = 'data_MAJ2.csv'
    data_file_csv = data_store_path+data_name
    # modify the json name
    # => data_MAJ.json
    data_file_json = data_store_path + 'data_MAJ2.json'
    ########################################  
    comments = pd.read_csv(data_file_csv)

    data_json = comments.to_json(data_file_json)

    data = comments.to_json()
    
    return data


@app.post("/uploadfile/", name ='upload file', tags = ['Data Management'])
async def create_upload_file(file: UploadFile):
    """
    the uploaded file is saved with current datatime
    """

    ########################################
    # modify this adresse to airflow
    data_store_path = '/app/clean_data/'
    # modify this file to origin database
    # => data_MAJ.csv
    # REAL MAJ data base
    data_origin = 'data_MAJ2.csv'
    
    ######################################## 

    print(file.content_type)
    
    data_file = data_store_path + data_origin
    dt = datetime.datetime.now()
    # if the hour and minute cause traitement problem, we can keep only y-m-d
    time = dt.strftime('%Y-%m-%d-%H:%M')

    file.filename = 'new_data_'+ time + '.csv'
    if file.content_type == "text/csv": 
        # write into a new data file
        with open(data_store_path + file.filename,'wb') as f:
            f.write(await file.read())

        # update the data_MAJ file (global data base)
        added_data = pd.read_csv(data_store_path + file.filename)
        added_data.to_csv(data_file, mode = 'a',header = False, index = False)
        
        # modify the ID
        data = pd.read_csv(data_file)
        data['inID'] = range(len(data))
        data.to_csv(data_file,index = False)
        return {"filename": file.filename}
    else:
    # elif file.content_type =='application/json':
        return {'INFO':'please use a csv file'}
   
