# NLP. Processing and sentiment analysis
The <a href="https://www.kaggle.com/utkarshxy/stock-markettweets-lexicon-data">dataset</a> is about tweets collected between April 9 and July 16, 2020. These tweets are related to the stock market and were obtained by using not only the SPX500 tag but also the top 25 companies in the index and "#stocks". It contains a total of 5000 tweets but only 1300 were manually classified and reviewed and 4 attributes.
- id: contains id used for the tweet
- created_at: date and time when the tweet was tweeted
- text: tweet/text written by the user
- sentiment: whether the tweet was positive, neutral, or negative

The goal of this assignment is to make a data processing and analysis to extract some useful knowledge and create a classification model to decide the sentiment of future tweets.

### Libraries required
The data analysis has been developed using the following software:
- R language 4.1.2
- tm 0.7-8
- textclean 0.9.3
- ggplot2 3.3.5
- ggwordcloud 0.5.0
- tensorflow 2.7.0
- keras 2.7.0
- kableExtra 1.3.4
- utf8 1.2.2
- spacyr 1.2.1

In order to install TensorFlow, follow the instructions <a href="https://tensorflow.rstudio.com/installation/">here</a> to install it in your platform.

### How to Run the Project
For the execution of this code you should follow this steps:
1. Install and import all the packages above
2. Set the working directory to the one that contains the dataset
3. If you have never used spacyr before, run the following command
```r
spacy_install() #  This will install a miniconda environment
```
