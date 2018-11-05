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


#----------------------------------------------
# Plot dos dados

#----------------------------------------------
# Ajuste do modelo
fit = lm(tempo_reacao ~ idade,
         data = df)
# Resumo
summary(fit)

# Anova
anova(fit)


#----------------------------------------------
# Estudo dos erros

# Add os erros no conjuntos de dados
df$erros = fit$residuals

# Add no conjunto de dados, os erros normalizados
df$erros_normalizados = erros_normalizados(df$erros) #Utilizando a função do outro arquivo de código

# Add a classificação dos erros em termos de sua variabilidade 
df$classe_erros = erros_outliers(df$erros)

# Histograma dos erros
ggplot(df, aes(x = erros_normalizados, color = classe_erros)) +
  geom_histogram(aes(y=..density..), colour = 'black') 
#  geom_density(alpha=0.2, fill = "#FF6666")

names(df)

