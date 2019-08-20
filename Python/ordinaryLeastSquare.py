#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
    LINEAR REGRESSION
"""
"""----------------------------------------------------------------------------
    Importando as libraries
"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt


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



"""----------------------------------------------------------------------------
    Preparando a base para a regressão e estimando a regressão
"""
y = df["mpg"]
X = df[['cylinders', 'displacement', 'weight','acceleration', 'model_year', 'origin']]


regression = sm.OLS(y, X).fit()
print(regression.summary())


#Plotando o gráfico
x = [x for x in range(len(X))]
prstd, iv_l, iv_u = wls_prediction_std(regression)

fig, ax = plt.subplots(figsize=(14,12))
ax.plot(x, y, 'o', label="data")
ax.plot(x, y, 'b-', label="True")
ax.plot(x, regression.fittedvalues, 'r--.', label="OLS")
#ax.plot(x, iv_u, 'r--')
#ax.plot(x, iv_l, 'r--')
plt.title('Title', loc='left', fontsize=20)
plt.xlabel('X axis')
plt.ylabel('Y axis')
ax.legend(loc='best')





