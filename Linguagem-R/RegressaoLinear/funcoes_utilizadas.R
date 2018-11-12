#----------------------------------------------
# Funções utilizadas *
#----------------------------------------------


# função de normalização dos erros
erros_normalizados = function(erros) {
  return((erros - mean(erros))/sd(erros))
}


# função de padronização dos erros
erros_padronizados = function(erros) {
  return(erros/sd(erros))
}


# Função que verifica os outliers
erros_outliers = function(erros){
  
  # Padronizando os erros
  erros_padronizados = erros_normalizados(erros) #Reaproveitando a função erros_normalizados
  
  # Criando um Data Frame com os erros
  df_erros = data.frame('erros_padronizados' = erros_padronizados)
  
  # Criando a classe dos erros
  df_erros$classe_dos_erros = '< 1 dp'
  df_erros$classe_dos_erros[erros_padronizados < qnorm(1-0.9544) | erros_padronizados > qnorm(0.9544)] = '< 2 dp'
  df_erros$classe_dos_erros[erros_padronizados < qnorm(1-0.9974) | erros_padronizados > qnorm(0.9974)] = '< 3 dp'
  
  # Exportando o resultado em forma de vetor
  return(df_erros$classe_dos_erros)
  
}


# Função que plota o gráfico de scatter plot
scatter_plot = function(xvar, erros){
  
  #Normalização dos erros
  erros_normalizados = erros_normalizados(erros)
  
  #Criação do data frame
  df = data.frame('xvar'= xvar, 'erros'=erros_normalizados)
  
  #Gráfico
  ggplot(df, aes(x = xvar, y = erros))+
    geom_point(shape=1)+
    geom_smooth(method=lm)+
    labs(title = 'Scatter plot dos erros')+
    xlab('Variável de entrada')+
    ylab('Erros')+
    ylim(c(min(erros)*1.2, max(erros)*1.2))
}


# qqplot dos erros
gg_qplot=function(erros){
  
  #Normalização dos erros
  erros_normalizados = erros_normalizados(erros)
  
  #Estruturação do data frame
  df = data.frame('erros_normalizados'=erros_normalizados)
  
  #Plot do gr?fico
  ggplot(data = df, mapping = aes(sample = erros_normalizados)) +
    stat_qq_point() +
    stat_qq_band(alpha = 0.5) +
    stat_qq_line() +
    #Títulos
    ggtitle('QQ-Plot dos erros')+
    xlab('Quantis amostrais')+ylab('Quantis te?ricos')
  
}


# Histograma dos erros
gg_hist = function(erros, classe_erros, bins = 30){
  
  #Estruturação do data frame
  df = data.frame('erros'=erros,'classe_erros'=classe_erros)
  
  #----PLOT DO HISTOGRAMA
  ggplot(data=df, aes(x=erros, fill=classe_erros)) + 
    geom_histogram(color = 'black', bins = bins)+
    scale_color_manual(values=c('cornsilk','cornsilk3','darkgoldenrod1'))+
    scale_fill_manual(values=c('cornsilk','cornsilk3','darkgoldenrod1'))+
    geom_vline(aes(xintercept = 0), colour='black', linetype='dotted')+
    #Títulos
    ggtitle('Histograma dos erros')+
    xlab('Erros')+ylab('Frequ?ncia abs')+
    #Edição do gráfico
    scale_y_continuous(labels = scales::comma)+
    scale_x_continuous(breaks = seq(min(erros)*1.3, max(erros)*1.3, 5))+
    theme(plot.title = element_text(size = 14, face = 'bold'),
          axis.title = element_text(size = 12),
          axis.text  = element_text(size = 11),
          legend.position='bottom',legend.title=element_text(size=9),
          legend.text=element_text(size=10))
  
}