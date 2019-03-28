# @ Neeraj Tripathi

# Sentiment Analysis of Movie Reviews Dataset (polarity dataset v2.0)
# Source : http://www.cs.cornell.edu/people/pabo/movie-review-data/

# Importing the libraries
import numpy as np
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.datasets import load_files

dataset = load_files('dataset/')
X,y = dataset.data, dataset.target

""" For large no  of reviews pickle X and y after loading once 
    and then load them from pickle file
    
with open("X.pickle","wb") as f:
    pickle.dump(X, f)

with open("y.pickle","wb") as f:
    pickle.dump(y, f)

with open("X.pickle","rb") as f:
    X = pickle.load(f)

with open("y.pickle","rb") as f:
    y = pickle.load(f)
"""

# PreProcessing the reviews

cleaned_reviews = []
#lemmatizer = WordNetLemmatizer()
for x in X :
    x = str(x).replace(r"\n"," ")       #x = re.sub(r"\n"," ", str(x)) Not working
    # Try replacing words like don't, won't, ain't, couldn't ..... properly
    x = re.sub(r"\W", " ", x)
    x = x.lower()
    x = re.sub(r"\d", " ", x)
    x = re.sub(r"\s[a-zA-Z]\s", " ", x)
    x = re.sub(r"^[a-zA-Z]\s+", "", x)
    x = re.sub(r"\s+$", "", x)
    x = re.sub(r"^\s+", "", x)
    x = re.sub(r"\s+", " ", x)
    #words = nltk.word_tokenize(x)
    #newWords = [word for word in words if word not in stopwords.words('english')]
    #newWords = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    #cleaned_reviews.append(' '.join(newWords))
    cleaned_reviews.append(x)
    
# Creating Tfidf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.6, min_df=50, stop_words=stopwords.words("english"))
cleaned_reviews = vectorizer.fit_transform(cleaned_reviews).toarray()
