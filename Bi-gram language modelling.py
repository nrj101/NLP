import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Read the review text only. Labels not needed because it's not a classification task.
reviews_dataset = pd.read_csv('Restaurant_Reviews.tsv', sep='\t').drop(['Liked'], axis=1)

# Preprocessing the data
dataset = ""
for i in range(len(reviews_dataset)):
    sentence =reviews_dataset.iloc[i, 0].lower()
    sentence = re.sub(r"n't", "n not", sentence)
    sentence = re.sub(r"let's", "let us", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"i ain't", "i am not", sentence)
    sentence = re.sub(r"he ain't", "he is not", sentence)
    sentence = re.sub(r"ain't", "are not", sentence)
    sentence = re.sub(r"\W", " ", sentence)
    sentence = re.sub(r"\d", " ", sentence)
    sentence = re.sub(r"\s[a]\s", " atat ", sentence)
    sentence = re.sub(r"\s[i]\s", " itit ", sentence)
    sentence = re.sub(r"\s[a-z]\s", " ", sentence)
    sentence = re.sub(r"atat", "a", sentence)
    sentence = re.sub(r"itit", "i", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"^\s", "", sentence) 
    sentence = re.sub(r"\s$", "", sentence) 
    sentence = re.sub(r"\.+",".", sentence)
    if i==0:
        dataset += sentence + "."
    else:
        dataset += " " + sentence + "."
        
sentences = nltk.sent_tokenize(dataset)

######################  Bigram model  ######################

# 1. Histogram of occurences of words in the dataset
def get_word2count(sentences):
    word2count = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1            
    return word2count
word2count = get_word2count(sentences)
print(len(word2count))

import heapq
freq_words = heapq.nlargest(1500, word2count, key=word2count.get)  # Select only 1500 most frequent words

# 2. Mapping most frequent words in the dataset to a unique index. This is used to translate sentences consisting of words to a list of numbers (indexes).
def get_word2idx_and_idx2word(sentences, freq_words):
    word2idx, idx2word = {}, []
    word2idx['START'] = 0
    word2idx['END'] = 1
    idx2word.append('START')
    idx2word.append('END')
    i = 2
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            if word not in word2idx.keys() and word in freq_words:
                word2idx[word] = i
                idx2word.append(word)
                i += 1
    word2idx['UNKNOWN'] = i
    idx2word.append('UNKNOWN')            
    return word2idx, idx2word

word2idx, idx2word = get_word2idx_and_idx2word(sentences, freq_words)
print(len(word2idx))

# 3. Transforming sentences from a collection of "word strings" to their corresponding indexes i.e. obtaining equivalent numerical representation of sentences.
def transform_sentences(sentences, word2idx):
    new_sentences = []
    for sentence in sentences:
        new_sentence = [word2idx['START']]
        for word in nltk.word_tokenize(sentence.lower()):
            if word not in word2idx.keys():
                #print(word)
                new_sentence += [word2idx['UNKNOWN']]
            else:
                new_sentence += [word2idx[word]]
        new_sentence += [word2idx['END']]
        new_sentences.append(new_sentence)
    return new_sentences

new_sentences = transform_sentences(sentences, word2idx)

# 4. Learn probability distribution of bi-grams in the above dataset
def get_bigram_probs(sentences, word2idx):
    V = len(word2idx)
    bigram_matrix = np.ones((V,V))              # not initializing with zeros so that there is some minimalistic probability for all bigrams.
    for sentence in sentences:
        for i in range(1, len(sentence)):
            bigram_matrix[sentence[i-1], sentence[i]] += 1
    bigram_matrix /= bigram_matrix.sum(axis=1, keepdims=True)
    return bigram_matrix

bigram_probs = get_bigram_probs(new_sentences, word2idx)

# 5. Calculate score corresponding to each sentence
def get_sentence_score(sentence, bigram_probs):
    score = 0
    for i in range(1, len(sentence)):
        score += np.log(bigram_probs[sentence[i-1], sentence[i]])
        
    return score / len(sentence)

# 6. Transform a sentence back from numerical representation to words.
def inv_transform_sentence(sentence, idx2word):
    new_sentence = ""
    for idx in sentence:
        word = idx2word[idx]
        if word in ['START','END']:
            continue
        if len(new_sentence)<1:
            new_sentence += word
        else:
            new_sentence += " "+word
        
    return new_sentence



# Test on a sentence from the dataset
print("Sentence: {}  Score: {}".format(inv_transform_sentence(new_sentences[34], idx2word), get_sentence_score(new_sentences[34], bigram_probs)) )


# Test on a correct new sentence
test_sentence = transform_sentences(["This is a delicious recipe from a popular north Indian restaurant."], word2idx)
print("Sentence: {}  Score: {}".format(inv_transform_sentence(test_sentence[0], idx2word), get_sentence_score(test_sentence[0], bigram_probs)) )

# Test on an incorrect new sentence
test_sentence = transform_sentences(["kind Food of like I what morning the in"], word2idx)
print("Sentence: {}  Score: {}".format(inv_transform_sentence(test_sentence[0], idx2word), get_sentence_score(test_sentence[0], bigram_probs)) )

# Add the .("dot") at the end of the above sentence to see the score improving a bit.
test_sentence = transform_sentences(["kind Food of like I what morning the in."], word2idx)
print("Sentence: {}  Score: {}".format(inv_transform_sentence(test_sentence[0], idx2word), get_sentence_score(test_sentence[0], bigram_probs)) )



# 7. Generate a sentence from initial word(s) using this bigram model
def generate_sentence(init_sentence, bigram_probs, word2idx, idx2word, max_len=20):
    try:
        last_word = word2idx[init_sentence.split(" ")[-1].lower()]
    except:
        print("The last word you entered could not be found in the corpus. Aborting execution !!")
        return init_sentence
    
    generated_idxs = []
    for i in range(max_len):
        next_idx = np.random.choice(np.arange(len(bigram_probs)), 1, p=bigram_probs[last_word, :])[0]   # choose next word using the probability-distribution learned in the bigram-model
        
        if idx2word[next_idx]=="END":
            break
        else:
            generated_idxs.append(next_idx)
    generated_words = inv_transform_sentence(generated_idxs, idx2word) 
    return init_sentence + " " + generated_words

print(generate_sentence("I have visited", bigram_probs, word2idx, idx2word, max_len=20) )
print(generate_sentence("Food", bigram_probs, word2idx, idx2word, max_len=20) )
print(generate_sentence("potato", bigram_probs, word2idx, idx2word, max_len=20) )