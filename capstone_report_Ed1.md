# Machine Learning Engineer Nanodegree
## Capstone Project
George Seah 
July 15th, 2017

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

#### Domain Background of Machine Learining in manufacturing testing.
The proposal domain is in the manufacturing testing field. In most of the mass production manufacturing, testing are part of the manufacturing process which help to ensure product quality and reliability. At the same time, testing also involve higher cost to the manufacturer and time consuming. 
The proposed project is to examine the prediction of the testing time required based on the all the available features. The proposed project is based on the Kaggle competitions: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/
We will use the data from this competition to examine different machine learning method in making prediction.



### Problem Statement


The problem we tried to solve is to predict the test time (the 'y') required based on all the featured provides (total 376 features).


#### Measurement 
Based on the competition request, we found out that the prediction is scored based on R^2 value (Coefficient of Determination), so we will use the same metric to score and compare our model

#### Anonimyzation of the dataset column name and data processing
As the dataset column name are anonymized, so we wouldn't be able to know the underlying meaning of each variable. It creates some issues for us to understand well the data, such as if we have done the Principal Component Analysis, we wouldn't be able to know what the first few component would actually mean.
After some research in the community discussion, one of the way that many data science expert use in this competition are adding all the principal component and their original component as part of the features set and select the algorithm, such as gradient boosting or random forest or regularized regression like Lasso that would select through the features set that make best prediction. 

#### Potential solution

The potential solution would be using gradient boosting regression tree. During the gradient boosting regression tree model building, I explore two different transformation --label encoding for all the multi-categorical variables.
I also built a random forest as a benchmark model to compare the performance of gradient boosting.
Besides, I also built a stacked model that combine lasso, gradient boosting and random forest for comparison.
Apart from using the cross-validation R^2 score from our code, I also use to Kaggle submission to cross-check the performance of the model.




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

For the coming discussion, I will focus 3 of the most common metrics -- MAE , MSE, and R^2

1. Mean Absolute Error (MAE) - 
	
	It takes absolute on the difference between predicted value and actual value. The key benefit of MAE would be interpretation as it measure the average eror across all the prediction.
	Shortfall of MAE is that it is using absolute which make it mathematically not a mathematically differentiable function as compared to Mean Squared Error (MSE)
	
2. Mean Squared Error (MSE) -

	It take squared value on the differences between predicted value and actual value. There are two key benefit would be the function is differentiable, which is very helpful if we use any Machine Learning Model utilized gradient descent to tune the model.
	Another feature of MSE is that it penalized on higher error since it take squared operation on the differences.

3. R^2<sub>2</sup> -
	
	It capture the propotion of variance that explained by the model. One of the key feature of R^2 is that it could be negative value. It means that the total predicted error from the model are higher than the total variance of the data.

	
Since we are mainly focus on boosting or ensemble method, all 3 of them should be feasible for the modelling.If we plan to use any model that are using gradient descent algorithm such as Neural Network, I think MSE is more appropriate as it mathematically differentiable.


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

The dataset provided from the competition has total 376 features. all of them are cateogorical. There are 8 variables are multi-categorical, where X4 has the smallest category count at 4 and X0 has the highest category count at 49.

<Fig 1. add picture of the table of count distinct category >

As we can see from Fig 2. below, all features are anonymized, so we wouldn't be able to know what would be the underlying meaning of each feature.

<Fig 2. Add a picture of the visualization of the table set of data>

Besides, we also found some cateogorical variable has only one value. 

<Fig 3. Add a visual of the constant variable>

The datasets has split into two separate sets -- training set and test set. Eact set of data has total 4209 datapoint.



### Exploratory Visualization and Analysis

#### Start from the "y"
We start by look at the y by plotting against the ID

<Fig 4. plot ID vs y>

From the data, there is no obvious relationship between the ID and "Y".


#### Looking at the "x"

From the "x", we start by looking at the 8 multi-categorical variables. As we can see from the figure below, most of them are highly skewed towards few categories except X8.

We then look at rest of the binary variables, we also found some variables have only one value.

Our first intuition from such observation is that some of the variables that has only one categorical value or very small observations can be removed from the data to reduce the dimensionality.
As those variables least likely will have any value in training the models since they only has one value across the full spectrum of "y".

#### Additional feature extractions


### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_


Due to anonymized independent variable, the ensemble method such as xgboost or random forest would be ideal in such situation since we couldnt make any assumption (or bias) behind the data.
So, we would need to find a algorithm that could help us select the important variables.


#### Algorithms and Techniques
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
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

The gradient boosting implementation procedure as following:
1. Load the data into jupyter notebook
2. Pre-processing based on the different approach we want to test.
3. Import requested ML library, such as Xgboost
4. use gradient boosting cross validation method to find out the optimal number of tree based on R^2 score
5. Run the gradient boosting one more time based on the number of trees selected

The stacked model implementation procedure as following:
1. Load the data into jupyter notebook
2. Pre-processing based on the different approach we want to test.
3. Import requested ML library, such as Xgboost,Gradient Boosting Regressor, Lasso, RandomForestRegressor, lightgbm
4. Prepare helper function to implement the sub-models
5. Use xgboost as aggregator model to tune the final result of all the sub-model

We also run the benchmark model - Random Forest, the implementation as following:
1. Load the data into jupyter notebook
2. Pre-processing the data based on the final model approach selected
3. Import requested ML library, such as RandomForestRegressor
4. use gridsearchCV to tune the hyperparameter of Random Forest
5. Run the random forest one more time based on final hyper-parameter selected



### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_

#### Initial solution
My initial solution is to use xgboost with one-hot encoding for all the variables and excluding the ID. 
The rationale is that ID should be just an indexing of the each data row and should not be part of the predicting variable.

After running xgboost optimized with number of trees, the result is not good.
Following are the score for the initial model:
| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| Xgb with OHE 	|  |  |  |

#### Subsequent solution testing  -- Xgboost with label encoding
Follow on the initial solution, I tried to use label encoding on the dataset with ICA and PCA. 

The result as following:
| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| Xgb with LE 	|  |  |  |

<Add scatter plot -- actual vs predicted>

As we can see from above, it significantly improve the R^2 score.
From the above results, one of the possible reason that we would see such improvement is that the category within each variables seems to have ordinality and thus explain the improvement in predictability.

#### Subsequent solution testing -- Stacked model
The stacked model is combining various model for the predictions.

The result as following:
The result as following:
| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| Stacked Model |  |  |  |

One of the interesting observation of the stacked model is that we get better result in both cross validation and public leaderboard scores, but not the private leaderboard.
From this observation, we mostl likely have already slightly over-fitted the data as compare to the xgboost.

<Add scatter plot -- actual vs predicted>




## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

Based on the cross validation scores, we selected the xgboost with label encoding as the final model as it has the highest R^2 performance scores.
It has the highest private leaderboard score, indicating that it has better generalization on the unseen data



### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_

#### Benchmark model -- Random Forest
We run the benchmark model - Random Forest. The results shows that 
| Model Name 	| Cross Validation Score | Public Leaderboard | Private Leaderboard |
| Random Forest |  |  |  |

<Add scatter plot -- actual vs predicted>


Comparing our final model results as compare to the benchmark result, we can see that our final selected model performed relatively well compare to the benchmark model.

Based on the current private leaderboard No 1 team results at ____, The final selected model is significant as a starting point model to know what would be the estimate test time would be given the results.

From the data, we've seen one outlier data point we extreme high test time. Based on the unsupervised learning, we found out that it 

Further investigation in such outlier would help us to unveil insight in optimizing the model and the operational issue behind it.


## V. Conclusion
_(approx. 1-2 pages)_


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


### Improvement

During the implementation and research, I found one of the new exploratory data analysis method called t-SNE(t-distributed Stochastic Neighbor Embedding) <sup>6</sup>, which can help us to visualize high dimension data in lower dimension space.
I believe we could make further improvement in data pre-processing if we able to extract some pattern from the data via this method.
Besides, Another approach that I think that may worth to test is deep learning. I believe it could be work compare its result versus xgboost.




-----------

### Footnotes
1. [Scikit-learn Regresion Metrics]http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
2. [Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
3. [XGboost Parameters] (https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)
4. [Why Stacked Model Perform Effective Collective Classfication] (https://kdl.cs.umass.edu/papers/fast-jensen-icdm2008.pdf)
5. [Stacked Regression] (http://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf)
6. [t-distributed Stochastic Neighbor Embedding Wikipedia] (https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
