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
 	Text: when modi promised “minimum government maximum governance” expected him begin the difficult job reforming the state why does take years get
	      justice state should and not business and should exit psus and temples
 	Target: -1


# Result

 

# References:
Millin, Oliver T., Jason C. Furtado, and Jeffrey B. Basara. "Characteristics, Evolution, and Formation of Cold Air Outbreaks in the Great Plains of the United States." Journal of Climate (2022): 1-37.

Rolnick, David, et al. "Tackling climate change with machine learning." ACM Computing Surveys (CSUR) 55.2 (2022): 1-96.
