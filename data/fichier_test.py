def total(liste):
    """ renvoie la somme des éléments d'une liste """
    result: float = 0.0
    for item in liste:
        result += item
    return (result)


def test_total():
    assert (total([1.0, 2.0, 3.0])) == 6.0


# executer $ pytest => ok ça marche !

