## NLP_project_nyu

# Authors
 
Changhyun Lee, Yunjeon Lee, Jennifer Rodriguez-Trujillo

# Research Question

Common works have attempted to tackle the concept of sentiment analysis in social media setting. Sentiment Analysis can be defined a process that, "automates mining of attitudes, opinions, views and emotions from text, speech, tweets and database sources through Natural Language Processing (NLP)" (Kharde,Sonawane, 5). Multitude of papers explore the different approaches one could possibly take when exploring sentiment analysis with the intent of targetting different end goals. In this particular paper,  the creators benchmark different models such as Naïve Bayes, SVM, and Maximum Entropy with Naïve Bayes setting the baseline model. The idea of these sentiment analysis models is to provide different outlets am methods of identifying this sentiment. More broadly spreaking, these can be applicable in: reviews from websites, applications as a sub-component technology, in business intelligence, domains, and even smart homes. Furthermore, we intend to replicate this model and adding an extension to it through the use of BERT or GPT-2.

# Background 
Precise electricity demand forecast is a crucial part of ensuring electric grid stability. Since the electric grid has limited capacity to store energy, inaccurate electricity demand forecast could lead to, at best, a waste of unused energy, and, at worst, grid failure that leads to significant economic and potential human loss. It is a task that has only become increasingly important in recent years. The extreme weather caused by climate change increases electricity demand fluctuations and puts more stress on the grid. The cold-air outbreak of 2021, for example, devastated Texas’ electric grid and left the entire state without light or heat for weeks (Millin). The introduction of weather-dependent renewable energy sources such as wind and solar also requires the grid operators to know more about the demand ahead of time. Precise short-term forecasts can allow operators to reduce their reliance on polluting standby coal power plants. Precise long-term forecasts can help system operators (and investors) to build more variable power sources such as wind and solar (Rolnick). For all the reasons mentioned above, any improvement an ML algorithm can create in accuracy or speed can create a significant societal impact.

# Project

In this project, we are using webscrapped Twitter data in order to examine the performance of Naïve Bayes, SVM, and Maximum Entropy alongside different n-grams .To further extend thi sexisting project, we will be using GPT-2 and/or Bert alongside benchmarking the three models presented.

# Data

The data we will use consists of daily electricity price and demand data between 1 January 2015 and 6 October 2020 (2016 days total) for Australia's second most populated state, Victoria. A brief description of the data is as follows:

	Text: a twitter post 
 Target: the values -1 represents negative sentiment, 1 represents positive sentiment, and 0 represents neutral
 
 EX. 
 Text: when modi promised “minimum government maximum governance” expected him begin the difficult job reforming the state why does take years get justice state should and not business and should exit psus and temples
 Target-1


# Result

We split the data we have into the following subsets: training data (Jan 1st, 2015 - Dec, 31st, 2017), validation data (Jan, 1st, 2018 - Dec. 31st, 2019), and test data (Jan, 1st 2019  - Dec, 31st 2019). As our baseline model, we used NeuralProphet without hyperparameter tuning. We built this model to examine the foundational components of NeuralProphet to compare against other existing models. The one factor we adjusted for within the baseline model is the use of the add_country_holidays function to add Australian holidays, which we also did for the tuned model. 
When running our baseline model on the training set, we found the Root Mean Square Error  (RMSE) to be 7578.40 and the Mean Average Error (MAE) to be 5592.61. 

After implementing our baseline model, we proceeded with finding our best hyperparameters. To best fit the needs and characteristics of the model, we use the following parameters to tune: num_hidden_layers, changepoints_range, n_lags, epochs, learning_rate, seasonality_reg, trend_reg, batch_size, and loss_func. We found the following values to have the best accuracy outcome:

	num_hidden_layers = 1
	changepoints_range = 0.975
	n_lags = 7, epochs = 175
	Learning_rate = 7
	Seasonality_reg = 0.15
	Trend_reg = 0.1
	batch_size = None
	loss_func = "MSE"

As a measure of accuracy to compare the values tested, we used RMSE and MAE (more information regarding the values can be found in the document). However, for reasons that we will later elaborate, this work is rendered mostly unusable by NeuralProphet’s limitations in hyperparameter tuning. The only parameter values we could specify were changepoint values, which are manually specified dates that prove to be outliers time and time again. To attain these values, we plotted the training data set and checked for high fluctuations. We measured the accuracy of this newly defined optimal model through the RMSE, which produced a value of 7382.88.

We then compared NeuralProphet to two classical models: ARIMA and exponential smoothing. The main reason why ARIMA was chosen was its popularity in forecasting. One key component of ARIMA is that the data must be stationary. The model is composed of three parts: 

	1) A weighted sum of lagged values of the series (Auto-Regressive(AR))
	2) A weighted sum of lagged forecasted errors of the series (Moving-Average part)
	3) A difference of the time series (Integrated(I))

In short, to build the model, we must verify that the data is trend stationary. In order to confirm this assumption, we ran a KPSS hypothesis test. More specifically, “trend stationary” assumes that each point is independent (i.i.d) of one another. We ran a KPSS significance test to confirm that the data is in fact trend stationary (test statistic is 0.061 and has a p-value of 0.100). There are three main parameters in ARIMA: P, D, and, Q. P is determined from the strongest correlation for ACF-Logged Data; D is the number of times that the raw observations are differenced, and Q is defined as the size of the moving average window, also called the order of moving average. We found that our best parameters for the model consisted of:

	P : lag =1 or lag = 7 
	D : 0 
	Q : 1 

All in all, to measure the accuracy of the model, we used RMSE, where the model produced an RMSE of 12265.4. 

The last model we used to benchmark NeuralProphet was Exponential Smoothing, a forecasting method for univariate time series data. This method produces forecasts that are weighted averages of past observations where the weights of older observations exponentially decrease. We found that the RMSE value to be 11709.3723. 

Ultimately, both the baseline and the tuned Neuroprophet models outperform classical methods. The two models’ accuracies exceed that of ARIMA and Exponential smoothing by about 38% and 35% respectively, as measured by Root Mean Square Error (RMSE). Between the two versions of Neuroprophet, the tuned model is slightly more accurate than the baseline, but the difference is not significant (less than 3 percent). 

One main source of struggle during this project is the problems that arise when one tries to implement NeuralProphet. NeuralProphet is a new package, hence it has limited support. When some of the model’s functions did not work, we often had trouble finding solutions to our bugs while seeking help in the developer community. Furthermore, it was also not clear to us at the start that one cannot manually specify most of the hyper-parameters of NeuralProphet, and that by attempting to do so, the model failed terribly. This challenge rendered most of our work in hyperparameter tuning unusable. We recommend Facebook AI to broaden users’ ability to specify hyperparameters.

In terms of the next steps, we recommend making context-based adjustments to the Neuralprophet model. For example, in electricity demand forecasting, the cost of error is not evenly distributed. In particular, under-forecasting could result in electricity shortage, which has significant economic and human costs. On the other hand, over-forecasting results in the loss of overproduced electricity, which is a minor cost in comparison. To make the model more applicable in energy demand forecasts, one could penalize the model for under-forecasting through parameter adjustment and feature engineering. Such a model should be evaluated not just based on accuracy, but also on a measure of the overall cost of errors.  

# References:
Millin, Oliver T., Jason C. Furtado, and Jeffrey B. Basara. "Characteristics, Evolution, and Formation of Cold Air Outbreaks in the Great Plains of the United States." Journal of Climate (2022): 1-37.

Rolnick, David, et al. "Tackling climate change with machine learning." ACM Computing Surveys (CSUR) 55.2 (2022): 1-96.
