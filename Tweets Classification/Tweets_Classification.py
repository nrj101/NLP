# @ Neeraj Tripathi

import tweepy
import pickle
from TweetsPreprocessor import TweetsPreprocessor
from tweepy import AppAuthHandler

# Initialising authentication keys and tokens
import auth_variables

consumer_key = auth_variables.consumer_key
consumer_secret = auth_variables.consumer_key_secret
access_token = auth_variables.access_token
access_secret = auth_variables.access_token_secret

# Using the API
auth = AppAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_secret)         #For OAuthHandler
search_keywords = ['twitter']
api = tweepy.API(auth_handler=auth, timeout=10)


tweets = []
""" To load previously collected tweets
with open("SampleTweets.pickle","rb") as f:
    tweets = pickle.load(f)
"""

# Fetching Tweets
query = search_keywords[0]
while(len(tweets)<200) :
    search_results = api.search(q=query+"-filter:retweets", lang='en', tweet_mode='extended', result_type='recent', count=100)
    for status in search_results :
        tweets.append(status.full_text)
    tweets = set([tweet for tweet in tweets])
    tweets = [tweet for tweet in tweets]

"""with open("SampleTweets.pickle","wb") as f:          # To save the collected tweets
    pickle.dump(tweets, f)"""

# Preprocessing the fetched tweets   
cleaner = TweetsPreprocessor()
cleaned_tweets = cleaner.clean_text(tweets)


# Loading the trained Sentiment Analysis model and the tfidf vectorizer
with open("Model.pickle","rb") as f:
    model = pickle.load(f)
    
with open("Vectorizer.pickle","rb") as f:
    vectorizer = pickle.load(f)

# Fitting and predicting   
X = vectorizer.transform(tweets).toarray()
y_pred = model.predict(X)


# Plotting the results
import matplotlib.pyplot as plt
import numpy as np
categories = ['Positive', 'Negative']
y_pos = np.arange(len(categories))

plt.bar(y_pos, [np.sum(y_pred==1), np.sum(y_pred==0)], alpha=0.5, color='magenta')
plt.xticks(y_pos, categories)
plt.ylabel("Number")
plt.title("Sentiment of tweets containing keyword : " + query)
plt.show()