#----------------------------------------------
# Regressão Linear Conceitos
#----------------------------------------------

#----------------------------------------------
# Suposiçoes
# 1) A(s) Variáveis de entrada estão controladas e, portanto, sob nenhum efeito externo
# 2) Os erros gerados pelo ajuste do modelo apresentam distribuição normal com média 0 e variância constante
# 4) Os erros não devem ser correlacionados

#----------------------------------------------
# Libraries utilizadas
library(ggplot2)
library(qqplotr)

#----------------------------------------------
# Separação dos dados em treinamento e teste
df$sample = sample(1:10, replace = TRUE, size = nrow(df))
treino = df[ df$sample <= 8,]
valida = df[!df$sample <= 8,]

#----------------------------------------------
# Plot dos dados

#----------------------------------------------
# Ajuste do modelo
fit = lm(yi ~ xi,
         data = treino)
# Resumo
summary(fit)

# Anova
anova(fit)

# Teste da regressão
valida$fitted = predict.lm(fit, valida)

# Plot dos dados


#----------------------------------------------
# Estudo dos erros

# Ajuste final do modelo
fit = lm(yi ~ xi,
         data = df)

# Summary do modelo
summary(fit)

# Add os erros no conjuntos de dados
df$erros = fit$residuals

# Add no conjunto de dados, os erros normalizados
df$erros_padronizados = erros_padronizados(df$erros) #Utilizando a função do outro arquivo de código

# Add a classificação dos erros em termos de sua variabilidade 
df$classe_erros = erros_outliers(df$erros)

# Scatter plot dos erros
scatter_plot(xvar = df$xi, erros = df$erros_padronizados)

# Ggplot dos erros
gg_qplot(df$erros)
