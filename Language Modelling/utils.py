import nltk
import numpy as np
import pandas as pd
import re

# 1. For preprocessing the data.
def preprocess(dataset):
    """Cleans & preprocesses the dataset for further use in NLP tasks.
    Arguments:
    dataset -- list, length N, each element containing sentence in natural text format.
    
    Returns:
    sentences -- list, length N, each element containing preprocessed output corresponding to each sentence of the input dataset."""
    sentences = []
    for i in range(len(dataset)):
        sentence = dataset[i].lower()
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
        sentence = re.sub(r"[\.]+","", sentence)
        
        sentences.append(sentence+".")
    return sentences

# 2. Histogram of occurences of words in the dataset
def get_word2count(sentences):
    """Generates the histogram for each word by counting the number of occurences across all "sentences".
    Arguments:
    sentences -- list, length N, each element containing preprocessed sentence of the dataset.
    
    Returns:
    word2count -- dictionary, with {word: # of occurences} as key:value pairs."""

    word2count = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence):
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1            
    return word2count

# 3. Mapping most frequent words in the sentences to a unique index. This is used to translate a sentence consisting of words to a list of numbers (indexes).
def get_word2idx_and_idx2word(sentences, freq_words):
    """Maps most frequent words across all "sentences" to a unique index, and generates a list that can reverse this mapping.
    Arguments:
    sentences -- list, length N, each element containing preprocessed sentence of the dataset.
    freq_words -- list, length V, each element is a word which is among top-V most frequent words in the "sentences".
    
    Returns:
    word2idx -- dictionary, with {word: mapped_index} as key:value pairs.
    idx2word -- list, length V, contains the corresponding word "token" at each index "idx", such that word2idx[token]=idx and idx2word[idx]=token. """
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

# 4. Transforming all sentences from a collection of "word strings" to their corresponding indexes i.e. obtaining equivalent numerical representation of sentences.
def transform_sentences(sentences, word2idx):
    """Convert each sentence in "sentences" from a collection of words to a collection of numbers, where a number is the unique index the original word was mapped to.
    Arguments:
    sentences -- list, length N, each element containing preprocessed sentence of the dataset.
    word2idx -- dictionary, with {word: mapped_index} as key:value pairs.
    
    Returns:
    new_sentences -- list, length N, each element containing the equivalent numerical representation of corresponding sentence in
                    sentences. This representation is obtained by mapping the words of each sentence to the mapped index."""

    new_sentences = []
    for sentence in sentences:
        new_sentence = [word2idx['START']]
        for word in nltk.word_tokenize(sentence.lower()):
            if word not in word2idx.keys():
                #print(word)
                new_sentence += [word2idx['UNKNOWN']]     # Replace with UNKNOWN token if the word in not among the most frequent words that were mapped to a unique index in word2idx.
            else:
                new_sentence += [word2idx[word]]
        new_sentence += [word2idx['END']]
        new_sentences.append(new_sentence)
    return new_sentences

# 5. Transform a sentence back from numerical representation to words.
def inv_transform_sentence(sentence, idx2word):
    """Convert the sentence from a collection of numbers back to a collection of words, where a word is obtained by reverse mapping the index through "idx2word".
    Arguments:
    sentence -- list, length w, each element represents a word in numerical format (specifically index number the word was mapped to).
    idx2word -- list, length V, contains the corresponding word "token" at each index "idx", such that word2idx[token]=idx and idx2word[idx]=token.   top-W most frequent words + ["START","END","UNKNOWN"]
    
    Returns:
    new_sentence -- list, length w, each element containing the actual "word-string"."""


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

# 6. Calculate score corresponding to a sentence using bigram probs obtained via counting method.
def get_sentence_score(sentence, bigram_probs):
    """Convert the sentence from a collection of numbers back to a collection of words, where a word is obtained by reverse mapping the index through "idx2word".
    Arguments:
    sentence -- list, length w, each element represents a word in numerical format (specifically index number the word was mapped to).
    bigram_probs -- numpy.ndarray, shape (V, V), contains the probability distributions for next word (along axis 1, i.e. columns) given current word (along axis 0, i.e. rows)
    
    Returns:
    score -- float, overall score that represents the log probability of the sentence among all possible sentences according to the model that learned "bigram-probs" distribution using counting method."""
    score = 0
    for i in range(1, len(sentence)):
        score += np.log(bigram_probs[sentence[i-1], sentence[i]])
        
    return score / len(sentence) # normalize by length of the sentence to avoid low scoring of longer sentences.


# 7. Generate a sentence from initial word(s) using counting method bigram model.
def generate_sentence(init_sentence, bigram_probs, word2idx, idx2word, max_len=20):
    """Complete a sentence given the initial words "init_sentence" using bigram language model obtained via counting method.
    Arguments:
    init_sentence -- string, first few words (empty string is allowed) of the sentence that is to be generated.
    bigram_probs -- numpy.ndarray, shape (V, V), contains the probability distributions for next word (along axis 1, i.e. columns) given current word (along axis 0, i.e. rows)
    word2idx -- dictionary, with {word: mapped_index} as key:value pairs.
    idx2word -- list, length V, contains the corresponding word "token" at each index "idx", such that word2idx[token]=idx and idx2word[idx]=token.   top-W most frequent words + ["START","END","UNKNOWN"]
    max_len -- int, maximum length (default 20 words excluding the words provided in "init_sentence") to consider while completing the sentence. The generated sentence may be shorter in length depending on whether "END" token was generated before reaching this maximum length.
    
    Returns:
    generated_sent -- string, sentence completed after "init_sentence" by using words generated by the counting-method bigram model."""
    try:
        last_word_idx = word2idx[init_sentence.split(" ")[-1].lower()]
    except:
        print("The last word you entered could not be found in the corpus. Aborting execution !!")
        return init_sentence
    
    generated_idxs = []
    for i in range(max_len):
        next_idx = np.random.choice(np.arange(len(bigram_probs)), 1, p=bigram_probs[last_word_idx, :])[0]   # choose next word using the probability-distribution learned in the bigram-model
        
        if idx2word[next_idx]=="END":
            break
        else:
            generated_idxs.append(next_idx)
    generated_sent = init_sentence + " " + inv_transform_sentence(generated_idxs, idx2word) 
    return generated_sent

# 8. Function for one-hot encoding the words: required for neural network models
def one_hot_encoding(idx, size=2000):
    """Converts word indexes to one-hot-encoded format.
    Arguments:
    idx -- int, number that represents the unique index the original word was mapped to.
    size -- int, the size of one-hot-encoding.
    Returns:
    vec -- numpy.ndarray shape(size, ), vector corresponding to one-hot-encoded idx."""
    vec = np.zeros((size, ))
    vec[idx] = 1
    return vec
    
    
# 9. Softmax function
def softmax(z):
    """Applies softmax function on the 2-D input matrix z, softmax is applied column-wise.
    Arguments:
    z -- numpy.ndarray shape(V, N), matrix containing elements over which softmax function is to be applied.
    Returns:
    s -- numpy.ndarray shape(V, N), matrix corresponding to operation softmax(z)."""

    z = z - z.max(axis=0)          # subract max of the column from each element of respective column for calculation stability. Results remain unaffected.
    
    s = np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)
    #print(z.shape, z.max(axis=0).shape, s.shape)
    return s

# 10. Calculate score corresponding to a sentence using weights of NN1 (no hidden layer) learned via back-propagation.
def get_sentence_score_nn1(sentence, weights):
    """Convert the sentence from a collection of numbers back to a collection of words, where a word is obtained by reverse mapping the index through "idx2word".
    Arguments:
    sentence -- list, length w, each element represents a word in numerical format (specifically index number the word was mapped to).
    weights -- numpy.ndarray, shape (V, V), contains the weights learned by the simple neural network model.
    
    Returns:
    score -- float, overall score that represents the log probability of the sentence among all possible sentences according to the model that learned "bigram-probs" distribution using counting method."""
    n = len(sentence)
    # convert each word of the sentence to one-hot-encoded format
    inp_sentence = np.zeros((n, weights.shape[0]))
    inp_sentence[np.arange(n), sentence] = 1  
    #print(np.argmax(inp_sentence, axis=1))       # to check if one-hot-encoding is correct
    
    X = inp_sentence[:n-1, :]              # shape: N x V
    
    # forward propagation
    preds = softmax(np.dot(weights, X.T))        # shape: V x N
    
    score = 0
    for i in range(n-1):
        score += np.log( preds[sentence[i], i] )        # sentence[i] denotes the idx corresponding to the word, preds[i] denotes the probability predicted by the model for each word in Vocabulary.
        
    return score / n # normalize by length of the sentence to avoid low scoring of longer sentences.

# 11. Generate a sentence from initial word(s) using NN1 (no hidden layer) bigram model.
def generate_sentence_nn1(init_sentence, weights, word2idx, idx2word, max_len=20):
    """Complete a sentence given the initial words "init_sentence" using bigram language model obtained via neural network with no hidden layers.
    Arguments:
    init_sentence -- string, first few words (empty string is allowed) of the sentence that is to be generated.
    weights -- numpy.ndarray, shape (V, V), contains the weights learned by the simple neural network model.
    word2idx -- dictionary, with {word: mapped_index} as key:value pairs.
    idx2word -- list, length V, contains the corresponding word "token" at each index "idx", such that word2idx[token]=idx and idx2word[idx]=token.   top-W most frequent words + ["START","END","UNKNOWN"]
    max_len -- int, maximum length (default 20 words excluding the words provided in "init_sentence") to consider while completing the sentence. The generated sentence may be shorter in length depending on whether "END" token was generated before reaching this maximum length.
    
    Returns:
    generated_sent -- string, sentence completed after "init_sentence" by using words generated by the NN1 bigram model."""
    try:
        last_word_idx = word2idx[init_sentence.split(" ")[-1].lower()]
    except:
        print("The last word you entered could not be found in the corpus. Aborting execution !!")
        return init_sentence
    
    generated_idxs = []
    for i in range(max_len):
        X = np.zeros((1, weights.shape[0]))
        X[0, last_word_idx] = 1
        
        # forward propagation
        probs = softmax(np.dot(weights, X.T))        # shape: V x 1
        
        next_idx = np.random.choice(np.arange(weights.shape[0]), 1, p=probs.ravel())[0]   # choose next word using the probability-distribution learned in the NN1 bigram-model
        
        if idx2word[next_idx]=="END":
            break
        else:
            generated_idxs.append(next_idx)
    generated_sent = init_sentence + " " + inv_transform_sentence(generated_idxs, idx2word) 
    return generated_sent


# 12. Calculate score corresponding to a sentence using weights of NN2 (one hidden layer) learned via back-propagation.
def get_sentence_score_nn2(sentence, layer1_weights, layer2_weights):
    """Convert the sentence from a collection of numbers back to a collection of words, where a word is obtained by reverse mapping the index through "idx2word".
    Arguments:
    sentence       -- list, length w, each element represents a word in numerical format (specifically index number the word was mapped to).
    layer1_weights -- numpy.ndarray, shape (D, V), contains the weights for first layer of the neural network model.
    layer2_weights -- numpy.ndarray, shape (V, D), contains the weights for second layer of the neural network model.
    
    Returns:
    score -- float, overall score that represents the log probability of the sentence among all possible sentences according to the model that learned "bigram-probs" distribution using counting method."""
    n = len(sentence)
    # convert each word of the sentence to one-hot-encoded format
    inp_sentence = np.zeros((n, layer1_weights.shape[1]))
    inp_sentence[np.arange(n), sentence] = 1  
    #print(np.argmax(inp_sentence, axis=1))       # to check if one-hot-encoding is correct
    
    X = inp_sentence[:n-1, :].T              # shape: V x 1
    
    # Forward propagation
    z1 = np.dot(layer1_weights, X)                     # shape: D x 1
    a1 = z1 * (z1>0)                       # shape: D x 1
        
    z2 = np.dot(layer2_weights, a1)                    # shape: V x 1
    preds = softmax(z2)                    # shape: V x 1
    
    score = 0
    for i in range(n-1):
        score += np.log( preds[sentence[i], i] )        # sentence[i] denotes the idx corresponding to the word, preds[i] denotes the probability predicted by the model for each word in Vocabulary.
        
    return score / n # normalize by length of the sentence to avoid low scoring of longer sentences.

# 13. Generate a sentence from initial word(s) using NN2 (one hidden layer) bigram model.
def generate_sentence_nn2(init_sentence, layer1_weights, layer2_weights, word2idx, idx2word, max_len=20):
    """Complete a sentence given the initial words "init_sentence" using bigram language model obtained via neural network with one hidden layers consisting of "D" units.
    Arguments:
    init_sentence -- string, first few words (empty string is allowed) of the sentence that is to be generated.
    layer1_weights -- numpy.ndarray, shape (D, V), contains the weights for first layer of the neural network model.
    layer2_weights -- numpy.ndarray, shape (V, D), contains the weights for second layer of the neural network model.
    word2idx -- dictionary, with {word: mapped_index} as key:value pairs.
    idx2word -- list, length V, contains the corresponding word "token" at each index "idx", such that word2idx[token]=idx and idx2word[idx]=token.   top-W most frequent words + ["START","END","UNKNOWN"]
    max_len -- int, maximum length (default 20 words excluding the words provided in "init_sentence") to consider while completing the sentence. The generated sentence may be shorter in length depending on whether "END" token was generated before reaching this maximum length.
    
    Returns:
    generated_sent -- string, sentence completed after "init_sentence" by using words generated by the NN2 bigram model."""
    try:
        last_word_idx = word2idx[init_sentence.split(" ")[-1].lower()]
    except:
        print("The last word you entered could not be found in the corpus. Aborting execution !!")
        return init_sentence
    
    generated_idxs = []
    for i in range(max_len):
        X = np.zeros((layer1_weights.shape[1], 1))    # shape: V x 1
        X[last_word_idx, 0] = 1
        
        # Forward propagation
        z1 = np.dot(layer1_weights, X)                     # shape: D x 1
        a1 = z1 * (z1>0)                       # shape: D x 1

        z2 = np.dot(layer2_weights, a1)                    # shape: V x 1
        probs = softmax(z2)                    # shape: V x 1
        
        next_idx = np.random.choice(np.arange(layer1_weights.shape[1]), 1, p=probs.ravel())[0]   # choose next word using the probability-distribution learned in the NN2 bigram-model
        
        if idx2word[next_idx]=="END":
            break
        else:
            generated_idxs.append(next_idx)
    generated_sent = init_sentence + " " + inv_transform_sentence(generated_idxs, idx2word) 
    return generated_sent