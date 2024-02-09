# Industrial Copper Modelling

## About:

This project is about processing the large given dataset to predit the status of the material like Won or Lost using clasification model and Predict the selling price using a regression model. Python scripting is used in this project.

## Problem Statement

1.Exploring skewness and outliers in the dataset.
2.Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.
3.ML Regression model which predicts continuous variable ‘Selling_Price’.
4.ML Classification model which predicts Status: WON or LOST.
5.Creating a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost)

## Python Libraries Used

Sklearn
Pandas
Numpy
Matplotlib
Seaborn
Pickle
Streamlit

## Data Preprocessing and Model Building

The Given data may have many data missing, Having outliers, Having White spaces, Having null values and having noices and other process hindrances,
We need to check and clean the data befor processing for a Machine learning model or EDA. Lets see the preprocessing steps involved in this given dataset.

Step:1
In the finrst hand we need to understand the data. See how the data is distributed, How many dependant and independant variables available, And how many features are there for the model selection

Step:2
Select the important features and chack them for the skewness, noise and outliers using histplot, violin plot and boxplot. 

Step:3
Some data may not be evaluated or visually analysed using the plots, since it hase more noise we use log transformation to transform the data to be worked in the model.

Step:4
By using box plot we can assume that there is some outliers, We can use IQR method and robust methods to clean the outlies. Since we worked woth the outliers, skewness and noises we can prepare the training and testing datasets to prepare the ML model.

Step:5
There is lot of clasification and regression models are available in sklearn we have to select the best model for the predictiondi of status and selling price. Clasification model is used for catagorical data since predicting status won or lost is catagorical we use clasification model for status prediction depending on the high accuracy. We selecting the Random Forest clasifier is it has the highest accuracy score on training and testing data.

Step:6
As same as the clasification we use Random forest regressor for the selling price prediction, Since the data we have to predict is numerical we are using regression model. The important creteria in regression model is R2 vallue. Random forest regressor has the high R2 value here.

Step:7
After the model is successfully build we are storing the model in picle to use it when ever we need the prediction. Since the buildig and testing the model took longer time we are reducing the time consumtion on every prediction we choose to save the model in a pickle file.

## Streamlit App Building

We are Checking the models using the streamlit app. I have created a streamlit app to test and predict the status and the Seling price.

## Thank you.
