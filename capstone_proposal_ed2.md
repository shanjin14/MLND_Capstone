# Machine Learning Engineer Nanodegree
## Capstone Proposal
George Seah
Jun 14th, 2017

## Proposal
_(approx. 2-3 pages)_


### Domain Background

#### Domain Background of Machine Learining in manufacturing testing.
The proposal domain is in the manufacturing testing field. In most of the mass production manufacturing, testing are part of the manufacturing process which help to ensure product quality and reliability. At the same time, testing also involve higher cost to the manufacturer and time consuming. 
The proposed project is to examine the prediction of the testing time required based on the all the available features. The proposed project is based on the Kaggle competitions: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/
We will use the data from this competition to examine different machine learning method in making prediction.


#### Why choose this topic? 
I'm interested in how we can bring machine learning to help manufacturing to be more productive or simply better.


### Problem Statement
_(approx. 1 paragraph)_

The problem we tried to solved is to predict the test time (the 'y') required based on all the feature provided (total 376 features).

#### Measurement 
The prediction is scored based on R^2 value (Coefficient of Determination).

#### Replicability
As the data are relatively structured, the problem and solution can be reproduced in the model if we set the random seed.

#### Potential soltion
The potential solution would be using gradient boosting regression tree.
During the development, we will explore other model possiblity.


### Datasets and Inputs
_(approx. 2-3 paragraphs)_

The dataset is obtained from Kaggle competition - [Mercedez-Benz Greener Manufacturing](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)

The dataset has the testing time as the dependent variable (the "y") and 376 feature ( the "X").
All 376 features are categorical, while the dependent variable is continuous.
The provided dataset has anonymized the all the features. As such, we would need to perform some clustering analysis and other exploratory data analysis to understand more about all the features, such as correlation between features.

There are 4209 instance inside the datasets.


### Solution Statement
_(approx. 1 paragraph)_

As mentioned in previous session, the proposed solutions are using gradient boosting regession. 
Gradient boosting regression is one of the most popular library in Kaggle. Based on [empirical comparison of supervised learning algorithm](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.60.3232), it shows that gradient boosting tree is one of the best learning algorithm<sup>1</sup>

One of the key factors to mentioned in the solutions is the pre-processing of the data. As all the independent variables in the datasets are catgorical. We would need to do one-hot-encoding to all the variable. A good explaination of one-hot-encoding can be found [here](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)<sup>2</sup>.

Besides, since we are working with 376 features, we could consider feature engineering such as Principal Component Analysis(PCA)<sup>3</sup> to identify if have any principal component that we could use to reduce the feature space.

Another approach to explore is Multiple Correspondence Analysis (MCA)<sup>4</sup>. Based on one of the exploratory data analysis done by kaggler community, we could use the technique to identify the key components needed to capture most of the "y" variable's variance<sup>5</sup>.



### Benchmark Model
_(approximately 1-2 paragraphs)_

The proposed benchmark model would be random forest regression as the benchmark model. The random forest algorithm is a model that perform well in prediction yet with few hyperparamter to tune. So, it is easier to set it up and use it as a benchmark model.

The result would be measure by R^2 value as the same as the potential solution proposed in above section.


### Evaluation Metrics
_(approx. 1-2 paragraphs)_

The proposed metric is R^2 value. Based on the predicted value in test set to know the prediction capability of the model.


### Project Design
_(approx. 1 page)_

Project Design workflow 
I would split the project workflow as following :

1. Undestand data
Apply general exploratory data analysis to understand more about the data. EDA that will be done includes:
1. Descriptive statistic 
2. K Mean Clustering to know if have any natural grouping in the data

2. Fit the benchmark model
Fit and tune the proposed benchmark model -- random forest

3. Fit the prpposed model 
Fit and tune the propose model - Gradient Boosting Regression. Besides, I will explore other alternative method such as SVR, Stacking ensemble method.

4. Predict and Compare
Make prediction based on the studies in steps 3 mentioned above and make final recommendation.
Finally, based on the model, generate the final prediction.




-----------
### Footnotes
1. [What is better: gradient-boosted trees, or a random forest?](http://fastml.com/what-is-better-gradient-boosted-trees-or-random-forest/)
2. [What is One Hot Encoding and when is it used in data science?](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)
3. [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
4. [Multiple Correspondence Analysis](https://en.wikipedia.org/wiki/Multiple_correspondence_analysis)
5. [EDA and Visualisation](https://www.kaggle.io/svf/1281435/ded107c7ae67f91b9c31e9835cbaf549/__results__.html#new-multiple-correspondence-analysis-mca)



**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
