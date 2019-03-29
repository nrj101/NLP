#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 07:18:01 2019

@author: neeraj
"""
import re
class TextPreprocessor:
    def __init__(self):
        print("Module imported successfully")
    def clean_text(self,X):
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
        return cleaned_reviews