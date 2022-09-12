# Gesture Classification Using Machine Learning Algorithms

Surface electromyographic data (sEMG) recorded in multichannels can be used to infer the activation of individual muscle groups involved in certain actions. The sEMG-pattern, which represents the degree of contraction of a group of muscles, can then be linked to each individual movement. This, in turn, allows for the classification of sEMG-patterns and, finally, the creation of a human–machine interface based on sEMG recordings.

In order to come up with a model which will classify the type of gesture, I used various machine learning algorithms. In recent advancement in the field of machine learning, the approach of ensembling is one of the most effective techniques leveraged to obtain good classification as well as regression scores. In my approaches, I used one such method known as Random Forest which proved to be the best model.

## Dataset Description

A MYO Thalmic bracelet was used on a user's forearm and a PC with a Bluetooth receiver to record patterns. The bracelet had eight sensors evenly placed around the forearm that collect myographic signals at the same time. The signals are delivered to a PC via a Bluetooth interface. Raw EMG data from 36 people were used as they made a series of static hand gestures. The subject performs two series, each consisting of six (seven) fundamental motions. Each move lasted three seconds, with a three-second rest in between. In each column, there are approximately 40000-50000 recordings (30000 listed as
guaranteed).

The attributes in the raw data file are as follows:

1. Time - time in ms;
2. Channels 2 - 9 i.e., eight EMG channels of MYO Thalmic bracelet;
3. Class - the label of gestures:

There were basically 7 types of gestures. They are as follows.

1. Label 0 - unmarked data,
2. Label 1 - hand at rest,
3. Label 2 - hand clenched in a fist,
4. Label 3 - wrist flexion,
5. Label 4 – wrist extension,
6. Label 5 – radial deviations,
7. Label 6 - ulnar deviations,
8. Label 7 - extended palm (the gesture was not performed by all subjects).

## Exploratory Data Analysis and Visualization

1. Time Series Plots
2. Principal Component Analysis
3. Exponential Moving Average
4. Correlation Map

## Methodology

After thorough exploratory data analysis, it becomes evident that time series analysis is not going to work. So, I went ahead with the traditional machine learning algorithms for classifying the gestures. Consider a feature vector for a particular time step to be the combination of the values obtained from each of the 8 channels. The class label was used as the target for all the classification models. 

I experimented with the following models.

1. Bernoulli Naive Bayes
2. Gaussian Naive Bayes
3. K Nearest Neighbours
4. Neural Network
5. Decision Tree
6. Random Forest
7. Ada Boost

## Conclusion

In this assignment, we found various models that can effectively capture the gesture when leveraged on the sEMG signals. A thorough exploratory data analysis gives us an intuition of the data, and the scatter plot of PCA gives us the linearly inseparability of the data as well as the relevance of each of the channel information to the actual label. Furthermore, the correlation plot provide an insight that none of the channel features can be neglected due to low correlation among them. The Random Forest model proves to be robust in classifying the gestures. This is also validated from good accuracy and F1-score. For all the data sets, the final accuracy and F1 Score lies in the range of 97-99%. For the future work, thorough hyperparameter tuning can be carried out in order to make other models robust to the channel data. Moreover, time series models such as LSTM, GRU and ARIMA can be used to forecast the values of the channels.
