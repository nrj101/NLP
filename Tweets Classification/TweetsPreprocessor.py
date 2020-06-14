#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 07:18:01 2019

@author: neeraj
"""
import re
#import nltk
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
class TweetsPreprocessor:
    def __init__(self):
        print("Module imported successfully")
    def clean_text(self,list_of_string_text):
        cleaned_reviews = []
        #lemmatizer = WordNetLemmatizer()
        for x in list_of_string_text :
            x = str(x).replace(r"\n"," ")       #x = re.sub(r"\n"," ", str(x)) Not working
            # Try replacing words like don't, won't, ain't, couldn't ..... properly
            x = x.lower()
            x = re.sub(r"won't", "will not", x)
            x = re.sub(r"i ain't", "i am not", x)
            x = re.sub(r"he ain't", "he is not", x)
            x = re.sub(r"can't", "can not", x)
            x = re.sub(r"n't", " not", x)
            x = re.sub(r"let's", "let us", x)
            x = re.sub(r"'s ", " is ", x)
            x = re.sub(r"wanna", "want to", x)
            x = re.sub(r"gonna", "going to", x)
            x = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s+", " ", x)
            x = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", x)
            x = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s+", " ", x)
            x = re.sub(r"\W", " ", x)
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
        return cleaned_reviews
