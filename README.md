# Predicting-Market-Returns-of-Hedge-Fund-Investments-Using-Machine-Learning-Algorithms

Introduction: 
Problem description:
Life in the financial market is defined by profit or loss. As a result, professionals and researchers always try to analyze the data to predict the market outcome. Here, we will be using data from Ubiquant investment (Beijing) Co., Ltd [1], which is a leading domestic quantitative hedge fund based in China. They rely on advanced machine learning algorithms and international talents to predict multiple stock prices in the Chinese market. Their main goal is to improve the ability of quantitative research to forecast better returns for their personal investments. In short, we are using time series investment data in order to predict target values to aid making investment decisions.

Motivation:
The main motivation behind this project is to gain insights about the chosen dataset and find underlying patterns from the anonymized variables that could help with creating a predictive model. The goal of the project is to predict the value (an obfuscated metric about investment's return rate) which is related to dozens of anonymous features. Furthermore, not only the names but values themselves might have been manipulated by the company as well. In the end, we built a model that forecasts an investment's return rate.We trained and tested our algorithm on historical prices and predicted the outcome, and used an accuracy metric (Minimum Absolute Error to be specific) measure our model’s performance .
In short, we are aiming to apply data mining concepts learned in the classroom to this practical problem in order to:
a ) Better understand data mining concepts.
b) Build a predictive model that can aid in personal investments.

Brief description of your report organization 
This report starts with the Data Exploration of our dataset and goes into the methods that were used to preprocess data. We then discuss the various models that we tested on this cleaned dataset . We also cover how we tested the performance of each of these models and explain the logic behind choosing our final model, considering all the parameters. 

Data Exploration:
•	Our data is continuous and not discrete. As a result we need to apply regression algorithms as opposed to classification algorithms in order to get the best results. 
•	Source: Active Kaggle competition (Ubiquant Market Prediction) [1]

Number of attributes/brief description of file:

•	Number of records: 31,414
•	Number of attributes: 304
It’s historic data that is extracted from thousands of real investments made by Ubiquant. Since the data is huge we will be using a parquet file loaded in kaggle to process it. [1]

<Investment Frequency(Randomly 20 Investment ID Selected) per Target>
•	X = Target (Stock Change)
•	Y = Investment Frequency(Randomly 20 Investment ID Selected
•	The single Investment ID has a lot of target values, since a single investment ID was used for a long period of time. The graph means the distribution of the target change from the 20 randomly selected investment ID.

Frequency of Investment per Time Series
o	X = Time ID(10 blocks of time period)
o	Y = Investment Frequency ()
o	 It describes the entire number of investments frequency during the 10 groups of time duration.

Data type & description for each attribute:
     df.describe() -> (time_id, investment_id, target, f_0, f_1) 

 
This only includes the certain data and not the whole as the description is too large to depict here. 

Missing values:
There are no missing values because of the nature of the dataset and the fact that it is for an active kaggle competition. It doesn’t require significant data cleaning in terms of filling in missing values. 

Each attribute/description for each attribute:
o	row_id - A unique identifier for the row.
o	time_id - The ID code for the time the data was gathered. The time IDs are in order, but the real time between the time IDs is not constant and will likely be shorter for the final private test set than in the training set.
o	investment_id - The ID code for an investment. Not all investments have data in all time IDs.
o	target - The target.
o	[f_0:f_299] - Anonymized features generated from market data.
Methodology 
Data Preprocessing:
o	Outlier Removal:We summed mean and standard deviation across the training dataset and multiplied it by a coefficient.We then removed all values that exceed a certain value (optimized to make training dataset normal).
o	Feature Selection:We dropped attributes row_id, time_id, investment_id i.e we dropped attributes not relevant to prediction
o	We also normalized the data by subtracting the mean (µ) of each feature and a division by the standard deviation (σ).
o	Normalization: We summed mean and standard deviation across the training dataset and multiplied it by a coefficient.We then removed all values that exceed a certain value (optimized to make training dataset normal).


Mining the Data 
Data Mining Techniques

•	Linear Regression
Linear regression is the prediction model that is based on supervised learning.  Given independent variables, the model predicts the value using the mean square error from the cost function. So, this regression technique finds out a linear relationship between x (input) and y(output).

•	Support Vector Regression
Support Vector Regression is the model that allows the regression model to be trained to satisfy a certain range of errors, rather than focusing on minimizing the sum of squared errors as an ordinary regression model does[2]. Our objective, when we are using SVR, is to basically consider the points that are within the decision boundary line. Our best fit line is the hyperplane that has a maximum number of points.
Parameterized values: epsilon: 0.1(0.1 epsilon error is allowed)

•	Bayesian Ridge Regression:
Bayesian regression allows a natural mechanism to survive insufficient data or poorly distributed data by formulating linear regression using probability distributors rather than point estimates. The output or response ‘y’ is assumed to be drawn from a probability distribution rather than estimated as a single value. One of the most useful types of Bayesian regression is Bayesian Ridge regression which estimates a probabilistic model of the regression problem.

•	Passive Aggressive Regression:
Passive-Aggressive algorithms are called so because :
Passive: If the prediction is correct, keep the model and do not make any changes. i.e., the data in the example is not enough to cause any changes in the model. 
Aggressive: If the prediction is incorrect, make changes to the model. i.e., some change to the model may correct it.
It’s one of the rare “online learning algorithms”. It useful in cases where the dataset is too large to train at once. The input comes sequentially and the machine learning model is updated step by step.

Models Performance  
Performance Comparison:

Motivation for Mean Absolute error [3] as a performance metric
We used percentage split to divide the dataset into 80% training to 20% testing set.
Then after the prediction for each model was done, we evaluated the accuracy of the model by using Mean Absolute Error (MAE), because our target value is continuous.

Performance/accuracy metric used was mean absolute error.
Formula for Mean Absolute Error [3]:

Yi = prediction
Xi = True Value
n = Total number of data points

For Python,
No.	Model	Mean Absolute Error
1	Support Vector Regression	0.683312209591877
2	Bayesian Ridge Regression	0.6834278841122793
3	Linear Regression	0.689403022554398
4	Passive Aggressive Regression	1.03279805571138

For Weka,
No.	Model	Mean Absolute Error
1	Linear Regression	0.6342
Lesser the Mean Absolute Error (MAE) [3], the better the model because it is the average difference between the predicted value and actual value. We used only Linear Regression in Weka because our GPU was too slow to do SVR for many records and other two python DM are not in Weka.

Conclusion:
In conclusion, Support Vector Regression[2] outperformed all the other models we tested in Python. While in Weka, it gave generally a better outcome than any of the models in python for Linear Regression. We achieved a reasonably low mean absolute error, which definitely puts us on the right track for building even more accurate models for personal investments. If we were to continue the project, we would focus on trying even more models and tuning them more in order to achieve a lower Mean Absolute error.


