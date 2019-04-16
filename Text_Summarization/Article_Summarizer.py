# @ Neeraj Tripathi

# Developing NLP model to produce a brief summary of large articles 

# Importing the libraries and packages
import bs4 as bs
from urllib import request
import re
import nltk
from nltk.corpus import stopwords
import heapq

# Getting article data from a url
source = request.urlopen('https://en.wikipedia.org/wiki/A._P._J._Abdul_Kalam').read()

# Parsing data
soup = bs.BeautifulSoup(source, 'lxml')

# Obtaining text data for input to the summarizer
article = []
for para in soup.find_all('p'):
    article.append(para.text)

article = ' '.join(article)
article = re.sub(r'\[[0-9]*\]', ' ', article)

# Preprocessing article data
words_only = article.lower()
words_only = re.sub(r'\(.*\)', ' ', words_only)
words_only = re.sub(r'\W', ' ', words_only)
words_only = re.sub(r'\d', ' ', words_only)
words_only = re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', words_only)
words_only = re.sub(r'^\s+', '', words_only)
words_only = re.sub(r'\s+$', '', words_only)
words_only = re.sub(r'\s+', ' ', words_only)


# Preparing histogram of words
stop_words = stopwords.words('english')
word_histogram = {}

for word in nltk.word_tokenize(words_only):
    if word not in stop_words:
        if word not in word_histogram.keys():
            word_histogram[word] = 1
        else:
            word_histogram[word] += 1

for key in word_histogram.keys():
    word_histogram[key] /= max(word_histogram.values())


# Function to compute average length of sentences in the article
def average_sentence_length(sentences_list):
    """Returns average length of sentences in the article"""
    avg = 0
    for sent in sentences_list:
        avg += len(nltk.word_tokenize(sent))
    avg /= len(sentences_list)
    return avg


# Model : Calculating sentence scores
sentences = nltk.sent_tokenize(article.lower())
avg_length = average_sentence_length(sentences)
sentence_scores = {}
for sentence in sentences:
    score = 0
    words = nltk.word_tokenize(sentence)
    
    if len(words) < avg_length + 5:
        for word in words:
            if word in word_histogram.keys():
                score += word_histogram[word]
        
        if sentence not in sentence_scores.keys():
            sentence_scores[sentence] = score
        else:
            sentence_scores[sentence] += score
            
# Picking top 10 sentences for summary          
summary = heapq.nlargest(10, sentence_scores, sentence_scores.get)
for i in range(0, len(summary)):
    summary[i] = summary[i].capitalize()
summary = ''.join(summary)