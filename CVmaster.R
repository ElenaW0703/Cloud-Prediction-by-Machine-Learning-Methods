

# this function takes a list, and merge all the elements into a matrix
mergelist = function(dlist, x=T){
  if(x == T){
    result = list()
  }else{
    result = c()
  }
  n = length(dlist)
  for(i in 1:n){
    # gets the element 
    if (x){
    element = dlist[[i]]
    result = rbind(result, element)}
    else{
      element =dlist[[i]]
      result = append(result, element)
    }
  }
  result = data.frame(result)
  return(result)
}

################################################################################

CVmaster = function(classifier, features, tr_labels, K,loss, setseed = FALSE){
  best = NULL
  if(!is.data.frame(features)){
  index = sample(1:K,size = length(tr_labels), replace = T)}
  else {index = sample(1:K,size = nrow(tr_labels), replace = T)}
  CV_loss = c()
  TR_loss = c()
  # set seed if needed
  if(setseed == T){
    set.seed(0)
  }
  feature_all = mergelist(features)
  label_all = mergelist(tr_labels, F)
  label_all$result = as.factor(label_all$result)
  data = cbind(feature_all, label_all)
  formula = paste(colnames(label_all),"~.",sep = "")
  formula = as.formula(formula)
  # tune parameter first using the entire data
  # tune for knn
  mod = classifier[["model"]]
  
  if(mod =="knn"){
    k_choices = classifier[["k"]]
    knn_tuned = tune.knn(x = feature_all,y = label_all[,1], k = k_choices)
    best_k = knn_tuned$best.parameters}
  
  # tune svm for polynomial kernel and gaussian kernel 
  else if(mod == "svm"){
    kernel_choices = classifier[["kernel"]]
    if (kernel_choices == "radial"){
      sigma_choices = classifier[["sigma"]]
      gamma_choices = classifier[["gamma"]]
      cost_choices = classifier[["cost"]]
      svm_tune = tune.svm(formula, data = data,
                          range = list(sigma = sigma_choices,
                                       gamma = gamma_choices,
                                       cost = cost_choices))
    }
    else if(kernel_choices == "polynomial"){
      degree_choices = classifier[["degree"]]
      coef0_choices = classifier[["coef0"]]
      cost_choices = classifier[["cost"]]
      gamma_choices = classifier[["gamma"]]
      svm_tune = tune.svm(formula, data = data,
                          range = list(degree = degree_choices,
                                       gamma = gamma_choices,
                                       cost = cost_choices,
                                       coef0 = coef0_choices))
    }}
  # tuning for random forest 
  else if(mod == "forest"){
    mtry_choices = classifier[["mtry"]]
    forest_tuned = tune.randomForest(x = feature_all,
                                     y = label_all[,1],
                                     mtry = mtry_choices)}
    
  # run the CV
  for(i in 1:K){
    # split data into training, validation, and testing 
    # if we use blocks, then the training and testing are lists
    # Check for one type since all the type of sets should be the same 
    if (!is.data.frame(features)){
    test_x = features[index == i]
    train_x = features[index!=i]
    test_y = tr_labels[index == i]
    train_y = tr_labels[index!= i]
    train_x = mergelist(train_x, TRUE)
    test_x = mergelist(test_x, TRUE)
    test_y = mergelist(test_y,FALSE)
    train_y = mergelist(train_y,FALSE)}
    else{
      test_x = features[index == i,]
      train_x = features[index!=i,]
      test_y = tr_labels[index == i]
      train_y = tr_labels[index!= i]
    }
    train_y[,1] = as.factor(train_y[,1])
    test_y[,1] = as.factor(test_y[,1])
    fit_data = cbind(train_x, train_y)
    formula = paste(colnames(train_y),"~.",sep = "")
    formula = as.formula(formula)
    # fit the classifier
    # classifiers that do not involve tuning parameters 
    if(mod == "logistic"){
      # no need to tune parameters 
      model = glm(formula, data = fit_data, family = "binomial")
      pred = predict(model,test_x, type = "response")
      pred = ifelse(pred>0.5, 1, -1)
      tr_fit = fitted(model)
      tr_fit = ifelse(tr_fit>0.5, 1, -1)
    }
    else if(mod == "qda"){
      model = qda(formula, data=fit_data)
      pred = predict(model,test_x)
      pred = pred$class
      tr_fit = predict(model, train_x)
      tr_fit = tr_fit$class
    }
    else if(mod =="lda"){
      model = lda(formula, data = fit_data)
      pred = predict(model, test_x)
      pred = pred$class
      tr_fit = predict(model, train_x)
      tr_fit = tr_fit$class
    }
    else if(mod == "naive bayes"){
      model = naiveBayes(formula, data = fit_data)
      pred = predict(model, test_x, type = "class")
      tr_fit = predict(model, train_x, type = "class")
      
    }
    # classifiers that involces tuning parameters
    else if(mod =="knn"){
      pred = knn(train_x, test_x, as.factor(train_y[,1]), k = best_k)
    }
    
    else if(mod == "svm"){
      kernel_choices = classifier[["kernel"]]
      if (kernel_choices == "radial"){
        best_cost = svm_tune$best.parameters[["cost"]]
        best_sigma = svm_tune$best.parameters[["sigma"]]
        best_gamma = svm_tune$best.parameters[["gamma"]]
        model = svm(formula, data = fit_data, cost = best_cost,
                    gamma = best_gamma,
                    sigma = best_sigma)
        best = list(cost = best_cost, 
                    sigma = best_sigma,
                    gamma = best_gamma)
      }
      else if(kernel_choices == "polynomial"){
        best_cost = svm_tune$best.parameters[["cost"]]
        best_degree = svm_tune$best.parameters[["degree"]]
        best_gamma = svm_tune$best.parameters[["gamma"]]
        model = svm(formula, data = fit_data, cost = best_cost,
                    gamma = best_gamma,
                    degree = best_degree)
        best = list(degree = best_degree,
                    cost = best_cost,
                    gamma = best_gamma)
      }
      pred = predict(model,test_x)
    }
    
    else if(mod == "tree"){
      model = tree(formula, data = fit_data)
      pred = predict(model, test_x, type = "class")
      tr_fit = predict(model, train_x, type = "class")
    }
    else if(mod == "forest"){
      best_mtry = forest_tune$best.parameters[["mtry"]]
      model = randomForest(formula,data = fit_data,
                           mtry = best_mtry)
      best = list(
                  mtry  = best_mtry)
      pred = predict(model,test_x)
    }
    if(loss =="Accuracy"){
      l = mean(pred!= test_y$result)
      tr_l = mean(tr_fit!= train_y$result)
    }
    else if(loss == "hinge"){
      l = max(0,1-t(pred)%*%test_y$result)
      tr_l = max(0,t(tr_fit)%*%train_y$result)
      
    }
    else if(loss == "exponential"){
      l = exp(-t(pred)%*%test_y$result)
      tr_l = exp(-t(tr_fit)%*%train_y$result)
    }
    CV_loss = c(CV_loss, l)
    TR_loss = c(TR_loss, tr_l)
  }
  returns = list(loss = CV_loss, best_model = model, best_para = best,
                 tr_loss = TR_loss)
}

