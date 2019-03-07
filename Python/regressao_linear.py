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
import pickle
import os


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



"""----------------------------------------------------------------------------
    Análise descritiva dos dados
"""
# Descritiva geral do dataset
df.describe()

# Avaliando a matriz de correlação dos dados
df.corr(method = 'pearson').round(3)

# Plotando um gráfico de linhas da variável resposta
fig, ax = plt.subplots(figsize=(15,4))
ax.set_title("Values", fontsize = 20)
ax.set_ylabel("Var Y", fontsize = 15)
ax.set_xlabel("Var X", fontsize = 15)
ax = df["mpg"].plot()

# Plotando um boxplot dos dados
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.set_ylabel("Var Y", fontsize = 10)
ax1.set_xlabel("Var X", fontsize = 10)
ax1.boxplot(df["mpg"])

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.set_ylabel("Var Y", fontsize = 10)
ax1.set_xlabel("Var X", fontsize = 10)
dados_filtrados = [df["mpg"][df["cylinders"] == 4], df["mpg"][df["cylinders"] == 3]]
ax1.boxplot(dados_filtrados)

# Plotando um histograma dos dados
plt.xlabel("Var")
plt.ylabel("Dist")
plt.title("Histograma dos dados")
plt.grid(True)
plt.hist(df["mpg"], density = True, alpha = 0.5, color = 'g')

# Plotando um Scatter plot
plt.xlabel("Var X")
plt.ylabel("Var Y")
plt.title("Scatter plot dos dados")
plt.grid(True)
plt.scatter(x = df["model_year"] ,y = df["mpg"])



"""----------------------------------------------------------------------------
    Preparando a base para a regressão e estimando a regressão
"""
y = df["mpg"]
X = df[['cylinders', 'displacement', 'weight','acceleration', 'model_year', 'origin']]

# Separando a amostra em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciando o modelo de regressão
modelo = LinearRegression()

# Ajustando o modelo aos dados
modelo.fit(X_train, y_train)



"""----------------------------------------------------------------------------
    Analisando os resultados
"""
labels_das_variaveis = ['Intercept','cylinders', 'displacement', 'weight', 'acceleration', 'model_year','origin']
betas = np.append(modelo.intercept_, modelo.coef_)

tabela_de_parametros = pd.DataFrame(data = betas, index = labels_das_variaveis, columns = ["parametros"])
del labels_das_variaveis, betas

# Print do R²
print("R² = {}".format(modelo.score(X_train, y_train).round(2)))

# Realizando a previsão
y_fitted = modelo.predict(X_test)
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
    Salvando o modelo estimado
"""
# Verifiando o diretório atual
print(os.getcwd())

# Alterando o diretório atual
os.chdir("/home/fabio/Documentos") 

# Salvando o modelo
output = open("modelo_regressao", "wb")
pickle.dump(modelo, output)
output.close()

# Importando o modelo
modelo2 =  open("modelo_regressao", "rb")
lm_new = pickle.load(modelo2)
modelo2.close()


