# @ Neeraj Tripathi

# Sentiment Analysis of Movie Reviews Dataset (polarity dataset v2.0)
# Data Source : http://www.cs.cornell.edu/people/pabo/movie-review-data/

# Importing the libraries
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.datasets import load_files

dataset = load_files('dataset/')
X,y = dataset.data, dataset.target

""" For large no  of reviews pickle X and y after loading once 
    and then load them from pickle file
# Pickling X and y    
with open("X.pickle","wb") as f:
    pickle.dump(X, f)

with open("y.pickle","wb") as f:
    pickle.dump(y, f)

# Unpickling X and y
with open("X.pickle","rb") as f:
    X = pickle.load(f)

with open("y.pickle","rb") as f:
    y = pickle.load(f)
"""

# PreProcessing the reviews
from TextPreprocessor import TextPreprocessor    
cleaner = TextPreprocessor()
cleaned_reviews = cleaner.clean_text(X) 


# Creating Tfidf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 3000,max_df=0.65, min_df=50, stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(cleaned_reviews).toarray()

# words = vectorizer.get_feature_names()

# Creating Train and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=272)


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# Training various machine learning models and predicting on train & test sets
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(random_state=0)
model1.fit(X_train, y_train)
cv_accuracies1 = cross_val_score(estimator = model1, X = X_train, y = y_train, cv = 10, n_jobs=-1)
y_pred1 = model1.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred1)

from sklearn.svm import SVC
model2 = SVC(kernel='linear', random_state=0)
model2.fit(X_train, y_train)
cv_accuracies2 = cross_val_score(estimator = model2, X = X_train, y = y_train, cv = 10, n_jobs=-1)
y_pred2 = model2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)


from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
model3.fit(X_train, y_train)
cv_accuracies3 = cross_val_score(estimator = model3, X = X_train, y = y_train, cv = 10, n_jobs=-1)
y_pred3 = model3.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred3)


"""
Results :           MODEL                   10 fold Cross_Validation Accuracy                 Test Set Accuracy

            Logistic Regression :                    0.8512               model1                    0.8575
            Support Vector Class. :                  0.8487               model2                    0.8300
            Random Forest Class. :                   0.8324               model3                    0.8275
"""


# Saving the Model and Vectorizer

with open("Model.pickle","wb") as f:
    pickle.dump(model1, f)
with open("Vectorizer.pickle","wb") as f:
    pickle.dump(vectorizer, f)