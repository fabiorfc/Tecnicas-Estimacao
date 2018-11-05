#===FUNÇÕES===#



#===FUNÇÃO PARA ADD UMA COLUNA ALEATORIA===#
k.sample = function(data, k = 5) {
  #Essa função recebe um data frame e adiciona um vetor aleatório com k categorias#
  data$sample = sample(1:k, nrow(data), replace = TRUE)
  return(data)
}


#===FUNÇÃO UTILIZAR O K-FOLD===#
k.fold_Reg.Log = function(data, k = 5) {
  #Essa função utiliza a amostragem k-fold para treinar e validar o modelo#
  #Leitura das bibliotecas necessárias
  
  library(pROC)
  #===FUNÇÃO PARA AMOSTRAGEM===#
  data = k.sample(data, k = k)
  
  #Iterador para calcular
  ks.vector = NULL; auc.vector = NULL
  for(i in 1:k){
    #Segmentação dos dados
    treino = data[ !data$sample == i , -ncol(data) ]
    teste  = data[  data$sample == i , -ncol(data) ]
    
    #Ajuste do modelo
    fit = glm(formula = TARGET ~ ., data = treino, family=binomial())
    
    #Predict do modelo
    teste$fitted = predict.glm(fit, teste)
    
    #Cálculo do KS
    options(warn=-1)
    ks = as.vector(ks.test(teste$fitted[teste$TARGET == 1], teste$fitted[teste$TARGET == 0])$statistic)
    options(warn=0)
    
    #Cálculo do AUC
    auc.Value = as.vector(roc(response = teste$TARGET, predictor = teste$fitted)$auc)
    
    #Concatenação dos vetores
    ks.vector  = c(ks.vector , ks)
    auc.vector = c(auc.vector, auc.Value)
    resultado = data.frame('ks' = ks.vector, 'auc' = auc.vector)
  } #Fim do loop de fit e predict
  
  return(resultado)
}