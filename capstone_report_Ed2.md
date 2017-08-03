# Machine Learning Engineer Nanodegree
## Capstone Project
George Seah 
July 15th, 2017

## I. Definition

### Project Overview

#### Domain Background of Machine Learining in manufacturing testing.
The proposal domain is in the manufacturing testing field. In most of the mass production manufacturing, testing are part of the manufacturing process which help to ensure product quality and reliability. At the same time, testing also involve higher cost to the manufacturer and time consuming. 
The proposed project is to examine the prediction of the testing time required based on the all the available features. The proposed project is based on the Kaggle competitions: [Mercedes-Benz Greener Manufacturing] (https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/) .
We will use the data from this competition to examine different machine learning method in making prediction.



### Problem Statement

The problem we tried to solve is to predict the test time (the 'y') required based on all the featured provides (total 376 features).

#### Measurement 
Based on the competition request, we found out that the prediction is scored based on R^2 value (Coefficient of Determination), so we will use the same metric to score and compare our model.

#### Anonimyzation of the dataset column name and data processing
As the dataset column name are anonymized, so we wouldn't be able to know the underlying meaning of each variable. It creates some issues for us to understand well the data, such as if we have done the Principal Component Analysis, we wouldn't be able to know what the first few component would actually mean.
After some research in the community discussion, one of the way that many data science expert use in this competition are adding all the principal component and their original component as part of the features set and select the algorithm, such as gradient boosting or random forest or regularized regression like Lasso that would select through the features set that make best prediction. 

#### Potential solution

The potential solution would be using gradient boosting regression tree. During the gradient boosting regression tree model building, I explore two different transformation --label encoding for all the multi-categorical variables.
I also built a random forest as a benchmark model to compare the performance of gradient boosting.
Besides, I also built a stacked model that combine lasso, gradient boosting and random forest for comparison.
Apart from using the cross-validation R^2 score from our code, I also use Kaggle submission to cross-check the performance of the model.


### Metrics

#### Selected Metric
Based on the competition request, the proposed metric is R^2. Based on the predicted value in test set to know the prediction capability of the model.
Although the competition mandatory to use R^2, but I think it is worthwile to discuss about different choices of the measurement metrics and why R^2 is suitable.

#### Choices of measurement metrics
Based on the research in sklearn metrics<sup>1</sup>, we can see that following are the list of metrics available:
1. Mean Absolute Error (MAE)
2. Mean Squared Error (MSE)
3. Median Absolute Erro(MedAE)
4. R^2
5. Explained Variance Score

For the coming discussion, I will focus the discussion on three of the most common metrics -- MAE , MSE, and R^2

1. Mean Absolute Error (MAE) - 
	
	It takes absolute on the difference between predicted value and actual value. The key benefit of MAE would be interpretation as it measure the average eror across all the prediction.
	Shortfall of MAE is that it is using absolute which make it mathematically not a mathematically differentiable function as compared to Mean Squared Error (MSE)
	
2. Mean Squared Error (MSE) -

	It take squared value on the differences between predicted value and actual value. There are two key benefit would be the function is differentiable, which is very helpful if we use any Machine Learning Model utilized gradient descent to tune the model.
	Another feature of MSE is that it penalized on higher error since it take squared operation on the differences.

3. R^2<sub>2</sup> -
	
	It capture the propotion of variance that explained by the model. One of the key feature of R^2 is that it could be negative value. It means that the total predicted error from the model are higher than the total variance of the data.

	
Since we are mainly focus on boosting or ensemble method, all 3 of them should be feasible for the modelling. If we plan to use any model that are using gradient descent algorithm such as Neural Network, I think MSE is more appropriate as it is mathematically differentiable.


## II. Analysis

### Data Exploration

The dataset provided from the competition has total 376 features. all of them are cateogorical. There are 8 variables are multi-categorical, where X4 has the smallest category count at 4 and X0 has the highest category count at 49.

|ColumnName |	TestDataUniqueValue |	TrainDataUniqueValue |
| ------------- |:-------------:| -----:|
|	X0 |	49 |	47 |
|	X1 |	27 |	27 |
|	X2 |	45 |	44 |
|	X3 |	7 |	7 |
|	X4 |	4 |	4 |
|	X5 |	32 |	29 |
|	X6 |	12 |	12 |
|	X8 |	25 |	25 |


<img src="http://instanas.com/gh_work/DataSample.JPG">

As we can see from table above, we can see that all features are anonymized.So, we wouldn't be able to know what would be the underlying meaning of each feature.

We plotted the multi-categorical variable frequency count, we can see that most of them are highly skewed towards few category with the extreme cases like X4, which has most of the sample with one value only.

<img src="http://instanas.com/gh_work/Freq/X0Freq.JPG">  <img src="http://instanas.com/gh_work/Freq/X1Freq.JPG">   
<img src="http://instanas.com/gh_work/Freq/X2Freq.JPG">  <img src="http://instanas.com/gh_work/Freq/X3Freq.JPG">   
<img src="http://instanas.com/gh_work/Freq/X4Freq.JPG">  <img src="http://instanas.com/gh_work/Freq/X5Freq.JPG">   
<img src="http://instanas.com/gh_work/Freq/X6Freq.JPG">  <img src="http://instanas.com/gh_work/Freq/X8Freq.JPG">   

<img src="http://instanas.com/gh_work/0_1_Count.png">
Count of binary value by each variable

For rest of the binary variables, we can see that we have some cateogorical variable has only one value. 

Finally, the datasets has split into two separate sets -- training set and test set. Eact set of data has total 4209 datapoint.



### Exploratory Visualization and Analysis

#### Start from the "y"
We start by look at the y by plotting against the ID

<img src="http://instanas.com/gh_work/YvsIDJointPlot.png">
Y vs ID joint plot

From the data, there is no obvious relationship between the ID and "Y". But, we can see that there is a downward trend as ID gets higher number.


#### Looking at the "x"

From the "x", we start by looking at the 8 multi-categorical variables. As we can see from the figure below, most of them are highly skewed towards few categories except X8.

We then look at rest of the binary variables, we also found some variables have only one value.

Our first intuition from such observation is that some of the variables that has only one categorical value or very small observations can be removed from the data to reduce the dimensionality.

As those variables least likely will have any value in training the models since they only has one value across the full spectrum of "y".

#### Additional feature engineering

Besides pre-processing the data set, we also applied various feature engineering technique to give more variable options to the model.
Feature engineering techniqued used includes:
1. Principal Component Analysis (PCA)
2. Independent Component Analysis (ICA)
3. Gaussian Random Projection
4. Sparse Random Projection
5. Singular Value Decomposition


### Algorithms and Techniques
Due to anonymized independent variable, the ensemble method such as xgboost or random forest would be ideal in such situation since we couldnt make any assumption (or bias) behind the data.
So, we would need to find a algorithm that could help us select the important variables.

#### Testing various combination of data pre-processing and models
During the model building, we explore different approach as followings:
1. Xgboost with one hot encoding of the multi-cateogorical variables. In addition, we run principal component analysis(PCA) and independent component analysis (ICA) and append the new variables into the datasets.
2. Xgboost with label encoding of the multi-categorical variables. In addition, we run PCA and ICA and append the new variables into the datasets.
3. Stacked model with label encoding:
	1. Gradient Boosting Regression
	2. Lasso Regression
	3. Light GBM
	4. Random Forest
	In addition, we run PCA, ICA,Singular Value Decomposition, Gaussian Random Projection and Sparse Random Projection to enrich the datasets
	
##### 1st approach -- Xgboost with one hot encoding 
From the dataset, the first thought that I have is to one hot encoding all the multi-categorical variables. The key rationale comes from the observation above that most of the multi-categorical variables are highly skewed. 
Although one hot encoding would increase the dimensionality of the dataset, but we could also exclude some of the category that has no data at all.
Most importantly, one hot encoding also has no asumption behind the ordinality of the category in each variables.

There are several hyperparameter that xgboost allow us to tuned,some of the key hyperparameter includes:
1. n_trees 		- number of trees
2. eta			- learning rate
3. max_depth	- maximum depth of a tree
4. subsample	- Ratio of the instance are used for training. 0.5 means 50% of the training set's instances
5. objective	- objective option tells the xgboost that whether we are working on regressin problem (reg:linear) or classification problem (reg:logistic). There are many other options available<sup>3</sup>

##### 2nd approach -- Xgboost with label encoding
On the contrary of above methods, we do label encoding on the multi-categorical variables. It means that we assume some sort of ordinality inside the categorical variables.
It would be a good model to run for comparing the result against the first approach

Same as above, we could tune several hyperparameter in Xgboost.

##### 3rd approach -- Stacked model with label encoding 
The 3rd approach is stacked model which we use a combination of different model to perform the prediction. 
There are several research paper showed that the stacked model could deliver a better prediction capability<sup>4 5</sup>.

#### Algorithm explained

In this section, we will explain the key models that we used in this report:
1. Gradient Boosting
2. Random Forest
3. Stacked Model

##### Gradient Boosting<sup> 6 7</sup>

Gradient boosting is an ensemble prediction method. It start by initializing the first classification or regression tree. Subsequently, based on the first tree prediction, we fit a second tree on the residual of the prediction (y_hat-y).
As we can continue to fit more tree on the residual, the model will get better in the prediction.

But, the question comes: How do gradient boosting avoid overfitting?

It use a technique called shrinkage. It is similar to learning rate in deep learning, which we multiply a learning rate between 0 and 1 on the residual that we created. As such, the models will slowly learn and improve the residuals.
From the research, it shows that it helps improve the model generalization capability significantly.

##### Random Forest<sup>8</sup>

Random Forest is another popular ensemble method in machine learning. It takes a different approach compare to gradient boosting.
Random Forest starts by construct multiple parallel trees. When building the branch in each tree, the models use a subset of variable for building the tree. 
Apart from that, it also use another technique called bagging (bootstrapping, replace or resampling). The technique is that each datapoint will get to resample more than one time for prediction.
Finally, it combines all the parallel trees as the final prediction.

As a result of the resampling and using subset of variables, it helps improve the model generalization capability.


##### Stacked model<sup>9</sup>

Stacked model is actually not a model or technique by itself. It is a technique of combining multiple models in a structural way to improve the prediction.
It starts building a set of base models such as gradient boosting , random forest or lasso regression. 
Each base model make its own prediction (Y_i) 
After that, we create a meta model such as gradient boosting or linear regression to combine them in the way that best improve the prediction.

Assuming a stacked model with 3 base model, the meta model of linear regression would be looks like:
y_final = sum(W_i*Y_i) + e


	
### Benchmark
Randomforest is used as the benchmark model to compare the model selected performed better or worse.

Randomforest is another popular decision tree ensemble method. It has better performance in preventing over-fitting compare to decision tree (single tree). 
It is relatively easy to tune the hyperparameter with python gridsearchCV. 

The primary rationale behind using Randomforest as the benchmark model is that:
1. It gives relative good performance in prediction compare to decision tree
2. It is similarly flexible compare to our primary method Xgboost


## III. Methodology

### Data Preprocessing

When building the model, we tried several data-processing approach to see what would be the prediction results:
1. One hot encoding all the variables.
2. Label encoding all the multi-categorical varibles
3. Remove variables that has only one value (either has 0 or 1 only)
4. Include the ID or exclude the ID has one of the variable
5. performed Principal Component Analysis, Independent Component Analysis, Gaussian Random Projection, and Sparse Random Projection
   -- The purpose of performing above variables transformation technique is to find out new synthesized variable that may have significance predicton capability.

### Implementation

The gradient boosting implementation procedure as following:
1. Load the data into jupyter notebook
2. Pre-processing based on the different approach we want to test. 
	1. I tried two different pre-processing approach here - label encoding and one hot encoding.
	1. After label encoding or one hot encoding, I implemented principal component analysis(PCA) and independent component analysis(ICA) to create new additional variable and appended them into the dataset. 
		So, the data will both the original "X" variable and also the newly-created synthesized variable based on PCA and ICA. 
3. Import requested ML library, such as Xgboost.
	1. If you have not install xgboost on windows(I am using windows) before, you may faced some challenge get it setup. Following are the good resource to make it work:
		[How to install Xgboost] (https://stackoverflow.com/a/39811079) -- look for answer from Brettlyman. It is the easiest one I found.
4. use gradient boosting cross validation method to find out the optimal number of tree based on R^2 score
	1. The main parameter I tuned in the gradient boosting is the num_boost_round by using cross validation. The num_boost_round tells the model how many trees is needed for best prediction based on cross validation.
	   We used the xgboost function xgboost.cv to tune the paramter.
5. Run the gradient boosting one more time based on the number of trees selected

The stacked model implementation procedure as following:
1. Load the data into jupyter notebook
2. Pre-processing based on the different approach we want to test.
	1. We applied label encoding on the data. After that, we appplied PCA, ICA, and other dimention reduction technique such as Gaussian Random Projection, Sparse Random Projection, and Singular Vector Decomposition(SVD) to enrich the predictor set.
3. Import requested ML library, such as Xgboost,Gradient Boosting Regressor, Lasso, RandomForestRegressor, lightgbm
4. Prepare helper function to implement the sub-models
	1. We wrote few helper function here to 
5. Use xgboost as aggregator model to tune the final result of all the sub-model

We also run the benchmark model - Random Forest, the implementation as following:
1. Load the data into jupyter notebook
2. Pre-processing the data based on the final model approach selected
3. Import requested ML library, such as RandomForestRegressor
4. use gridsearchCV to tune the hyperparameter of Random Forest
5. Run the random forest one more time based on final hyper-parameter selected

#### Key challenge in implementation
The key challenge in implement the approach is primarily at understanding models and the different parameter and its effect within each models. To reduce the hurdle in implementation, I spend some time to read through the documentation and online readings to understand about the approach.
Another good resources is public forum such as stackoverflow or kaggle forum, which I managed to find the answer I need to implement them.



### Refinement

#### Initial solution
My initial solution is to use xgboost with one-hot encoding for all the variables and excluding the ID. 
The rationale is that ID should be just an indexing of the each data row and should not be part of the predicting variable.

After running xgboost optimized with number of trees, the result is not good.
Following are the score for the initial model:

| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| ------------- |:-------------:| -----:| -----:|
| Xgb with OHE 	| 0.6365 | -0.48733 | -0.5714 |

<img src="http://instanas.com/gh_work/XGB_OHE.png">
Y_Actual vs Y_Predicted in train data


#### Subsequent solution testing  -- Xgboost with label encoding
Follow on the initial solution, I tried to use label encoding on the dataset with ICA and PCA. 

The result as following:

| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| ------------- |:-------------:| -----:| -----:|
| Xgb with LE 	| 0.6522 | 0.5648 | 0.5462 |

<img src="http://instanas.com/gh_work/XGB_LE.png">
Y_Actual vs Y_Predicted in train data

As we can see from above, it significantly improve the R^2 score.
From the above results, one of the possible reason that we would see such improvement is that the category within each variables seems to have ordinality and thus explain the improvement in predictability.

In both approach above, I mainly tuned the num of boosting round needed based on cross validation.


Following are the cross validation tuned results snapshot for xgboost with one hot encoding:
| Num of Boosting Round | Cross Validation R^2 scores |
| 50 | 0.0230|
| 100 | 0.377 |
| 200 | 0.507 |
| 500 | 0.606 |
| 600 | 0.621 |

The final selected number of boosting round is 604

Following are the cross validation tuned results snapshot for xgboost with label encoding:
| Num of Boosting Round | Cross Validation R^2 scores |
| 50 | 0.231 |
| 100 | 0.370 |
| 200 | 0.510 |
| 500 | 0.615 |
| 600 | 0.629 |


The final selected number of boosting round is 767


#### Subsequent solution testing -- Stacked model
The stacked model is combining various model for the predictions.

The result as following:

| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| ------------- |:-------------:| -----:| -----:|
| Stacked Model | 0.65 | 0.5767 | 0.5458 |

One of the interesting observation of the stacked model is that we get better result in both cross validation and public leaderboard scores, but not the private leaderboard.
From this observation, it may imply that we may have already over-fitted the data as compare to the xgboost.

<img src="http://instanas.com/gh_work/Stacked.png">

The key parameter tuning in the stacked model I did is in Random Forest, which I leveraged on the cross validation tuning I did in random forest for the model.
In the random forest tuning, following are the parameter I tuned using GridSearchCV :
1. "max_depth": [3, None]
2. "max_features": [1, 3, 6 , 10]
3. "min_samples_leaf": [1]
4. "bootstrap": [True]
5. "n_estimators":[1500,3000,5000,7000]

The final selected random parameter is :
| Parameter | Value |
| max_depth | None |
| max_features | 10 |
| min_samples_leaf |1 |
| bootstrap | True |
| n_estimators | 7000 |


## IV. Results

### Model Evaluation and Validation

Based on the cross validation scores, we selected the xgboost with label encoding as the final model as it has the highest R^2 performance scores.
It has the highest private leaderboard score, indicating that it has better generalization on the unseen data.
Besides, it is generally more simpler to undersand compare to stacked model.

The final configuration with xgboost with label encoding is based on the cross validation as mentioned in the refinement steps.
The parameter used in final model as following:
| Parameter | Value |
| max_depth | 4 |
| num_boost_round | 767 |
| eta | 0.005 |
| subsample | 0.95 |



### Justification

#### Benchmark model -- Random Forest
We run the benchmark model - Random Forest. The results shows that:

| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| ------------- |:-------------:| -----:| -----:|
| Random Forest | 0.92 | 0.4963 | 0.4468 |

<img src="http://instanas.com/gh_work/RF.png">
Y_Actual vs Y_Predicted in train data

#### Comparing to the benchmark model
Comparing our final model results as compare to the benchmark result, we can see that our final selected model performed relatively well compare to the benchmark model.

Based on the current private leaderboard No 1 team results at 0.5555, the final selected model is considered acceptable as a starting point model to know what would be the estimated test time would be given all the variables.

From the data, we've seen one outlier data point with extreme high test time. Further investigation in such outlier would help us to unveil insight in optimizing the model and the operational issue behind it.

One of the interesting observation is the Y_Actual vs Y_Predicted scatter plot for all 3 models vs benchmark model. We can see that our selected model has predicted lower value in "y" around 130 and above. Meanwhile, Random Forest seems do a better prediction on training set.
But, the end validation score in private leaderboard shows that the result is lower compare to gradient boosting. It means that the random forest may already have overfitted the data.


## V. Conclusion

### Free form Visualization 

#### Feature Importance Analysis 

Both gradient boosting and random forest will have the feature importance plot, which will tell us more about which feature contribute the most to the prediction.

Gradient Boosting feature importance plot

<img src = "http://instanas.com/gh_work/VarImp_XGB.png">

Random Forest feature importance plot

<img src = "http://instanas.com/gh_work/VarImp_RF.png">


As we can see from the plot above, X314 is one of the key feature in the data set. Besides, we also can see that the synthesized predictor from ICA and PCA actually contribute values to the model.
 



### Reflection

During this project, we've gone through the whole process of machine learning application :
1. Data visualization and understanding
2. Identify and implement data pre-processing
3. Identify models, test models, and implement them
4. Evaluate and select the final model 

We managed to improve the result from negative R^2 value to better result by evaluating different model approach.

The interesting and difficult aspects of the projects is that I get to learn more about the modelling method I am interested in such as gradient boost (with XGboost) and Stacked model.
Both of them are widely used in the kaggle community.
The other interesting aspect of this project is that all the independent variable are anonymized. It means that we would need to rely on statistical method to understand more about the data. Method that we used 
Yes, the final model fit my expectatiosn in terms of the prediction outcome and the overall implementation process can be generalized for similar prediction problems.

This is also my first time participating the kaggle competition. It has been a fruitful experience on how much I learned during the competition.



### Improvement

During the implementation and research, I found one of the new exploratory data analysis method called t-SNE(t-distributed Stochastic Neighbor Embedding) <sup>10</sup>, which can help us to visualize high dimension data in lower dimension space. I believe we could make further improvement in data pre-processing if we able to extract some pattern from the data via this method.

Other than that, I would think we could run unsupervised learning such as k-mean clustering to see if have any natural clustering pattern for those datapoints with high test time(high "y").

Besides, Another approach that I think that may worth to test is deep learning. I believe it would be worthwhile compare its result versus xgboost.




-----------

### Footnotes
1. [Scikit-learn Regresion Metrics]http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
2. [Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
3. [XGboost Parameters] (https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)
4. [Why Stacked Model Perform Effective Collective Classfication] (https://kdl.cs.umass.edu/papers/fast-jensen-icdm2008.pdf)
5. [Stacked Regression] (http://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)
6. [A Kaggle Master Explains Gradient Boosting] (http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
7. [Gradient Boosting Wikipedia] (https://en.wikipedia.org/wiki/Gradient_boosting)
8. [Random Forest Wikipedia] (https://en.wikipedia.org/wiki/Random_forest)
9. [Kaggle Ensembling Guide] (https://mlwave.com/kaggle-ensembling-guide/)
10. [t-distributed Stochastic Neighbor Embedding Wikipedia] (https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

Besides above references, the coding and development has referred a lot of public sharing in kaggle such as :

11. [Kaggle - Mercedes-Benz Greener Manufacturing Kernels] (https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/kernels)
   Kernel that referenced in the report and coding includes:
   1. https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes
   2. https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697
   3. https://www.kaggle.com/frednavruzov/baselines-to-start-with-lb-0-56
   4. https://www.kaggle.com/anokas/mercedes-eda-xgboost-starter-0-55

