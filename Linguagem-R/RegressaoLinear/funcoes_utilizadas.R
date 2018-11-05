#----------------------------------------------
# Funções utilizadas
#----------------------------------------------


# função de normalização
erros_normalizados = function(erros) {
  return((erros - mean(erros))/sd(erros))
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

