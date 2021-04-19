# Optimizing an ML Pipeline in Azure

## Overview
1Thiiis project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains client's data for a bank marketing campaign. The classification determines whether the client given the particular set of data on job, marital status and so on will (yes) or will not (no) subscribe to the product.

The best performing model Azure AutoML model using MaxAbsScaler XGBoostClassifier with an accuracy of 91.52%.

## Scikit-learn Pipeline
There are several main components to AzureML pipeline from data preparation to classification. The raw data ran through clean_data() function where null value rows were dropped and other features were cleaned and categorized. The model chosen was LogisticRegression. A hyperdrive configuration was created and fed in the parameter sampler, estimator, primary metric, concurrent runs and total runs. 20 runs ran in total (4 at a time) and the run with best accuracy was the best run. 
The parameter sampler chosen was RandomParameterSampling and the early termination policy chosen was BanditPolicy. The Random Parameter Sampling was applied with --C and max_iter as its paramters. The max_iter determines the number of iterations to train for in each run and the parameter is set at a choice between a range from 10-30. The --c is the inverse of regularization that determines a uniform value between 0 ans 10. The Bandit policy terminates those runs where the primary metric isnt within the specified slack factor compared to best performing run. The Bandit Policy has 1 as evaluation interval, 0.1 as slack factor and 5 as delay evaluation. The slack factor is th ratio that calculates the distance from the best performing experiment run. The evaluation interval determined the frequency at which the policy will be applied. Lastly, the delay evaluation determines the number of interval that is required to delay the first policy evaluation.

**What are the benefits of the parameter sampler you chose?**
Hyperparameters tuning is the process of finding the best configuration of hyperparamater that gives the best performance. The benefits are:
- Supports discrete and continuous hyperparameters
- Supports early termination of low-performance runs
- Can refine search space later for improvements
- Random selection of hyperparamaters value

**What are the benefits of the early stopping policy you chose?**
An early termination policy is used to allow the model to stop looking for the best performance run after certain number of failures. Using early stopping policy increases computational efficiency.

## AutoML
In most runs, Automl had the best model as VotingEnsemble but there was error in running the command 'run.getoutput()'. After upgrading some packages, the best model was then MaxAbsScaler XGBoostClassifier and the command 'run.getoutput()' ran successfully by showing the hyperparameters for the model.
Some of the hyperparamater involved are base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The best accuracy in Scikit-learn pipeline was of 90.6% while AutoML had 91.52%. The AutoML had better accuracy by around 1%.  The slight difference in accuracy could be due to the usage of early termination policy in  Scikit learn model.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Future improvement could involve testing out the Scikit learn model with a different early termination policy or parameter sampler that could allow the program to have more runs.
