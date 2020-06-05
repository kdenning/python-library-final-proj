#Accounting for dependencies:
import pandas as pd
import spacy
import en_core_web_md
nlp = en_core_web_md.load()  
import numpy as np


#Functions:
def get_vec(s:str) -> list:
    return nlp.vocab[s].vector.tolist()

def addv(x:list, y:list) -> list:
  result = [(c1 + c2) for c1,c2 in zip(x,y)]
  return result

def dividev(x:list, y) -> list:
  result = [(c1/y) for c1 in x]
  return result

def meanv(matrix: list) -> list:
    assert isinstance(matrix, list), f"matrix must be a list but instead is {type(x)}"
    assert len(matrix) >= 1, f'matrix must have at least one row'

    #Python transpose: sumv = [sum(col) for col in zip(*matrix)]

    sumv = matrix[0]  #use first row as starting point in "reduction" style
    for row in matrix[1:]:   #make sure start at row index 1 and not 0
      sumv = addv(sumv, row)
    mean = dividev(sumv, len(matrix))
    return mean

def sent2vec(sentence: str) -> list:

  matrix = []

  doc = nlp(sentence.lower())

  for i in range(len(doc)):
    token = doc[i]
    if token.is_alpha and not token.is_stop:
      vec = get_vec(token.text)
      matrix.append(vec)

  result = [0.0]*300
  if len(matrix) != 0:
    result = meanv(matrix)
  return result
