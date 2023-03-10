# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 01:41:01 2022

@author: user
"""
from fastapi import Depends, FastAPI, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext


app = FastAPI(
    title="Backend FastAPI by Quan, Éric & Fred",
    version="wikihappy.org"
)
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# dictionnaire des users avec hashage du mot de pass
users = {

    "alice": {
        "username": "alice",
        "hashed_password": pwd_context.hash('wonderland'),
    },

    "bob": {
        "username": "bob",
        "hashed_password": pwd_context.hash('builder'),
    },

    "clementine": {
        "username": "clementine",
        "hashed_password": pwd_context.hash('mandarine'),
    },
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash('4dm1N')  # 4dm1N

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


@app.get("/authentification", name="Hello", tags=['home'])
async def current_user(username: str = Depends(get_current_user)):
    """
    _summary_ : pour accéder à cette route, il faut au préalable que l'utilisateur se soit authentifié.

        HELLO user

    Args: username (str, optional): _description_. Defaults to Depends(get_current_user).

    Returns: _type_: string  _description_: affiche "Hello" + "username"
    """
    return "Hello {}".format(username)


# ----------------------------------2eme route Création de la base de donnée ------------------------------------------------ #


@app.post("/Postez un commentaire",
          name="Nouveau Commentaire ",
          tags=['admin'])
async def admin_user(comment: str, username: str = Depends(get_current_user)):
    """
    _summary_ : détails à donner


    Args: idem
    """

    # ? ajouter la date de saisie du commentaire en return
    return {username: comment}
