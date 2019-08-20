#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
    REGRESSÃO LINEAR
"""
"""----------------------------------------------------------------------------
    Importando as libraries
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split



"""----------------------------------------------------------------------------
    Carregando e tratando os dados
"""
# Faz a leitura dos dados
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", header = None, delim_whitespace = True)
# Renomeando as colunas do dataset
names = ["mpg","cylinders","displacement","horse_power","weight","acceleration","model_year","origin","car_name"]
df.columns = names
del names

# Removendo os valores ? dos dados
df['horse_power_tratado'] = df['horse_power'].apply(lambda x: None if x == "?" else x)
df['horse_power_tratado'] = df["horse_power_tratado"].convert_objects(convert_numeric = True)

y = df["mpg"]
X = df[['cylinders', 'displacement', 'weight','acceleration', 'model_year', 'origin']]



"""----------------------------------------------------------------------------
    AApply Recursive Feature Estimador
"""
regression = LinearRegression(fit_intercept=True)

selector = RFE(regression, n_features_to_select=len(X))
selector = selector.fit(X, y)

#Plot result
result = pd.DataFrame({'features':X.columns,
                      'get in (Y/N)':selector.support_,
                      'feature ranking':selector.ranking_})
print(result)



"""----------------------------------------------------------------------------
    AApply Recursive Feature Estimador many times
"""

def RFEIterator(X, y, estimator, split):
    
    support_vec, ranking_vec = [], []
    
    for i in range(7):
        
        X_train, _, y_train, _ = train_test_split(X, y, train_size=split, random_state=i*132 + 132)
        
        selector = RFE(estimator, n_features_to_select=len(X))
        selector = selector.fit(X_train, y_train)
        
        support_vec.append(selector.support_)
        ranking_vec.append(selector.ranking_)
    
    SupportTable = pd.DataFrame(support_vec, columns=X.columns)
    RankingTable = pd.DataFrame(ranking_vec, columns=X.columns)

    return SupportTable, RankingTable

regression = LinearRegression(fit_intercept=True)
RFEIterator(X, y, regression, 0.7)
