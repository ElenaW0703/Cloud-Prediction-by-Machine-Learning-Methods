To Reproduce purposes:

First read the three images and change the data name as the corresponding feature name. 
Draw point based on x and y coordinate in each of the image and color it by labels using ggplot
Merge the three images by row to create an entire data, so that we can calcualte percentages of labels 

Separate data into train and pred based on the label of the plot. Label != zero --> train, label == zero --> pred
Repeat these for image 1, 2, 3 and all. 
Then, draw the density plot for each feature in three image sets. 
Use ggplot to draw the pairwise plot for the selected feature CORR, NDAI, and SD, and color the points based on label. 
Now we split the data into blocks based on which method to use. 
1. split based on x, and y quanitles --> 16 folds per image
2. split based on x and y median  --> 4 folds per image 

########################################################################################################## \
Then we set the size of training, testing and validation size and use sample to draw those block indexes 
Use the block indexes to set up the training, testing and validation set. 

Then put each set into a data frame using and use the percentage of -1 in validation and testing as the trivial
classifier accuracy. Since it was not specified, setting training to predict 1 or -1 depends on personal definition
of a trivial classifier. 

Then fit a logistic regression on all the data including training, testing, and validation. 
Use the step function to estimate the variable importance. 
Fit a single feature classifier on the training set. 
In my example, I use the logistic regression, but I believe that there can be 
more diverse choices. Find the top 3 features with the highest testing accuracy. 
Then call the CVmaster.R 

Extract the feature and label with the block indexes from the list of block that we splitted. 

we need to define classifier which is a list. 
For CVmaster function, it takes a list, classifier, a feature set which can be a list or data frame, a label set, 
K, number of fold, loss function name, default as Accuracy, and setseed arguement default as True. 

For the classifier, the list should be formated as {model:'', parameters:abso}, the parameters should be whatever 
need to be tuned. There is not default value, so please put all the possible parameter in even there is no other choices 
to tune on. For example: {model ="svm", kernel = "radial", sigma = 1:2, cost = 1:2, gamma = 1:2}
The CV master would run a 10 fold CV on the feature and lable to tune the parameter, then it would run a k fold cv
using the tuned model to estimate loss. 
For the loss function, it can take accuracy, hinge, and exponential loss as argument, which are as following:
"Accuracy", "hinge", "exponential". 

The CVmaster would return the CV loss, CV training error, and the best parameter it tuned. If there is no parameter
need to tune such as qda and lda, it will return NULL. 

Notice: When tunning KNN, the function could return error in type of dependent variable, which would need to vectorized the labels
Notice: When splitting blocks is small, it could return dlist[[i]] out of bound for some unknown reason. Rerun that
chunk second time would work out and have no influence on the result. 

Then we fit the model on the entire training and validation set, and make prediction on the testing set.
Use the prediction result and pROC package, we can generate the ROC curve based the prediction. 
Notice: need to make prediction is in the probability format. Decision tree would require run model on 
dependent variable as unfactorized, that is to remove as.factor()

To calculate the F1 score, you would need MLmetrics package. 
Notice: The prediction should be back to class rather than probability. 

Model Reliance: 
shuffle the each feature at a time in the testing set, and estimate the loss in the testing accuracy. 
### Notice: Use copy to copy the testing set to avoid any destructive modification. Save the loss in a matrix, 
and plot the loss based on each variable name. 

Algorithm Reliance: 
Drop one feature on a copy of the training and testing set. Reestimate the model and the testing accuracy. 
Calculate the loss or difference and plot it by feature names. 

To find any patterns:
plot the density of features labeled based on whether that point is correctly classified or not. 

It is also to encourage to draw a new map and labeled similarly to check any regional pattern. 

########################################################################################################## \

repeat the procedure inside #############, for different splitting approaches. 
I do recommend to plot in a separate rmd file as *CVmaster* takes time to run. 
### *CloudData_Plots.Rmd* contains all the code to generate every single plot with caption 
### *CloudData_FinalCode.Rmd* contains all the code for this project
### *CloudDataPredictionProject_ElenaW.pdf* is the report for this project
