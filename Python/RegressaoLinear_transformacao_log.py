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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



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
df['horse_power_tratado'] = df['horse_power_tratado'].astype(float)
df = df[~df['horse_power_tratado'].isna()]



"""----------------------------------------------------------------------------
    Análises descritivas
"""
# Plotando um Scatter plot
plt.xlabel("Var X")
plt.ylabel("Var Y")
plt.title("Scatter plot dos dados")
plt.grid(True)
plt.scatter(x = df["horse_power_tratado"] ,y = df["mpg"])

# Plotando um histograma dos dados
plt.xlabel("Var")
plt.ylabel("Dist")
plt.title("Histograma dos dados")
plt.grid(True)
plt.hist(df["horse_power_tratado"], density = True, alpha = 0.5, color = 'g')

# Plotando um boxplot dos dados
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.set_ylabel("Var Y", fontsize = 10)
ax1.set_xlabel("Var X", fontsize = 10)
ax1.boxplot(df["horse_power_tratado"])



"""----------------------------------------------------------------------------
    Transformando os dados
"""
df["log_mpg"] = np.log(df["mpg"])
df["log_cylinders"] = np.log(df["cylinders"])
df["log_displacement"] = np.log(df["displacement"])
df["log_horse_power_tratado"] = np.log(df["horse_power_tratado"])
df["log_acceleration"] = np.log(df["acceleration"])
df["log_weight"] = np.log(df["weight"])

# Plotando um Scatter plot
plt.xlabel("Var X")
plt.ylabel("Var Y")
plt.title("Scatter plot dos dados")
plt.grid(True)
plt.scatter(x = df["log_horse_power_tratado"] ,y = df["mpg"])

# Plotando um histograma dos dados
plt.xlabel("Var")
plt.ylabel("Dist")
plt.title("Histograma dos dados")
plt.grid(True)
plt.hist(df["log_weight"], density = True, alpha = 0.5, color = 'g')

# Plotando um boxplot dos dados
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.set_ylabel("Var Y", fontsize = 10)
ax1.set_xlabel("Var X", fontsize = 10)
ax1.boxplot(df["horse_power_tratado"])



"""----------------------------------------------------------------------------
    Preparando a base para a regressão e estimando a regressão
"""
y = df["log_mpg"]
X = df[['log_cylinders', 'log_displacement','log_horse_power_tratado', 'log_acceleration','log_weight']]

# Separando a amostra em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciando o modelo de regressão
modelo = LinearRegression()

# Ajustando o modelo aos dados e fazendo a previsão
modelo.fit(X_train, y_train)
y_fitted = modelo.predict(X_test)


"""----------------------------------------------------------------------------
    Analisando os resultados
"""
labels_das_variaveis = ['intercept','log_cylinders', 'log_displacement','log_horse_power_tratado', 'log_acceleration','log_weight']
betas = np.append(np.exp(modelo.intercept_), np.exp(modelo.coef_))

tabela_de_parametros = pd.DataFrame(data = betas, index = labels_das_variaveis, columns = ["parametros"])
del labels_das_variaveis, betas

# Print do R²
print("R² = {}".format(modelo.score(X_train, y_train).round(2)))

# Realizando a previsão
print("R² = %s" % metrics.r2_score(y_test, y_fitted).round(2))

# Avaliação do EQM e REQM
EQM = metrics.mean_squared_error(y_true = y_test, y_pred = y_fitted).round(3)
REQM = np.sqrt(EQM).round(3)
print("EQM = {}, REQM = {}".format(EQM, REQM))



"""----------------------------------------------------------------------------
    Analisando as previsões do modelo
"""
# Plotando o scatter plot dos dados previstos com os dados observados
plt.xlabel("Valores previstos")
plt.ylabel("Valores observados")
plt.title("Scatter plot dos dados")
plt.grid(True)
plt.scatter(x = y_fitted, y = y_test)

# Avaliação dos resíduos
# Validação de homocedasticidades (variação constante)
residuos = y_fitted - y_test
plt.xlabel("Residuos")
plt.ylabel("Valores observados")
plt.title("Scatter plot dos residuos")
plt.grid(True)
plt.scatter(y = residuos, x = y_test)

plt.xlabel("Residuos²")
plt.ylabel("Valores observados")
plt.title("Scatter plot dos residuos²")
plt.grid(True)
plt.scatter(y = residuos**2, x = y_test)

plt.xlabel("Var")
plt.ylabel("Dist")
plt.title("Histograma dos dados")
plt.grid(True)
plt.hist(residuos, density = True, alpha = 0.5, color = 'g')



"""----------------------------------------------------------------------------
    Comparando os resultados das transformações com os reais
"""

pd.DataFrame({'log_y_fitted':y_fitted,
              'log_y_teste':y_test,
              'y_teste':np.exp(y_test),
              'y_fitted':np.exp(y_fitted)}).head()












"""
# Análise de correlação
df.corr()


plt.imshow(df.corr(), cmap="RdBu")
plt.show()


correlacao = df.corr()

fig, ax = plt.subplots()
im = ax.imshow(correlacao)

# We want to show all ticks...
ax.set_xticks(np.arange(df.shape[1]))
ax.set_yticks(np.arange(df.shape[1]))
# ... and label them with the respective list entries
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
"""

