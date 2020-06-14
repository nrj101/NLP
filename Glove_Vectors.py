import numpy as np
from sklearn.metrics import pairwise_distances
import time

# Load glove vectors
word2vec = {}
embedding, idx2word = [], []
with open('glove.6B/glove.6B.300d.txt','r') as f:
    for line in f:
        content = line.split()
        word = content[0]
        vec = np.asarray(content[1:], dtype='float32')
        word2vec[word] = vec
        idx2word.append(word)
        embedding.append(vec)
    embedding = np.asarray(embedding)
    print('# of Words (V): {}      Dimension (D): {}'.format(embedding.shape[0], embedding.shape[1]))
          
# Define distance function(s) to evaluate closeness of words given their vector representations
def euclidean_dist(vec1, vec2):
    return np.linalg.norm(vec1-vec2)

def cosine_dist(vec1, vec2):
    return 1 - (    np.dot(vec1, vec2)/  (  np.linalg.norm(vec1) * np.linalg.norm(vec2)  )    )

# Function to find the missing word in an analogy expression. Simple implementation using for-loop (slow).
def find_analogy_v1(word1, word2, word3, dist_fn, word2vec):
    """Returns a word 'word4' such that it follows the analogy   word1 - word2 = word3 - word4"""
    entry_time = time.time()
    try:
        vec1, vec2, vec3 = word2vec[word1], word2vec[word2], word2vec[word3]
    except:
        print('Error! One or more of ' + word1 + ' , '+ word2 + ' & '+ word3 + ' could not be found in the dictionary.')
    target_vec = vec3 - (vec1 - vec2) 
    
    min_dist, best_word = np.inf, ''
    for word in word2vec.keys():
        if word in [word1, word2, word3]:
            continue
        dist = dist_fn(target_vec, word2vec[word])
        if dist<min_dist:
            min_dist = dist
            best_word = word
    print(word1 + " - " + word2 + "  =  " + word3 + " - " + best_word + "            time: %.2fs"%(time.time()-entry_time))
    return best_word

# try to find out any biases in the embeddings
find_analogy_v1('king', 'man', 'queen', cosine_dist, word2vec)
find_analogy_v1('man', 'computer',  'woman', cosine_dist, word2vec)
find_analogy_v1('man', 'soldier',  'woman', cosine_dist, word2vec)
find_analogy_v1('woman', 'care',  'man', cosine_dist, word2vec)
find_analogy_v1('woman', 'nurse',  'man', cosine_dist, word2vec)
find_analogy_v1('woman', 'weak',  'man', cosine_dist, word2vec)
find_analogy_v1('man', 'intelligent',  'woman', euclidean_dist, word2vec)
find_analogy_v1('man', 'handsome',  'woman', cosine_dist, word2vec)
find_analogy_v1('woman', 'housewife',  'man', cosine_dist, word2vec)
find_analogy_v1('man', 'rich',  'woman', cosine_dist, word2vec)
find_analogy_v1('man', 'tall',  'woman', cosine_dist, word2vec)
find_analogy_v1('man', 'crime',  'woman', cosine_dist, word2vec)
find_analogy_v1('man', 'head',  'woman', cosine_dist, word2vec)
find_analogy_v1('man', 'leader',  'woman', cosine_dist, word2vec)
find_analogy_v1('black', 'crime',  'white', cosine_dist, word2vec)
find_analogy_v1('boy', 'science',  'girl', cosine_dist, word2vec)
find_analogy_v1('boy', 'study',  'girl', cosine_dist, word2vec)
find_analogy_v1('boy', 'thug',  'girl', cosine_dist, word2vec)
find_analogy_v1('black', 'prison',  'white', cosine_dist, word2vec)
find_analogy_v1('black', 'gangster',  'white', cosine_dist, word2vec)
find_analogy_v1('black', 'education',  'white', cosine_dist, word2vec)
find_analogy_v1('black', 'poor',  'white', cosine_dist, word2vec)
find_analogy_v1('black', 'rap',  'white', cosine_dist, word2vec)
find_analogy_v1('dog', 'love',  'cat', cosine_dist, word2vec)

# Function to find the missing word in an analogy expression. Vectorized implementation (faster).
def find_analogy_v2(word1, word2, word3, word2vec, idx2word, embedding, metric='cosine'):
    """Returns a word 'word4' such that it follows the analogy   word1 - word2 = word3 - word4"""
    entry_time = time.time()
    try:
        vec1, vec2, vec3 = word2vec[word1], word2vec[word2], word2vec[word3]
    except:
        print('Error! One or more of ' + word1 + ' , '+ word2 + ' & '+ word3 + ' could not be found in the dictionary.')
        return np.zeros(vec1.shape)
    target_vec = vec3 - (vec1 - vec2) 
    
    distances = pairwise_distances(X=target_vec.reshape(1, -1), Y=embedding, metric=metric, n_jobs=-1)
    best_word_idx = distances.argmin()
    best_word = idx2word[best_word_idx]
    print(word1 + " - " + word2 + "  =  " + word3 + " - " + best_word + "            time: %.2fs"%(time.time()-entry_time))
    return best_word

find_analogy_v2('dog', 'love',  'cat', word2vec, idx2word, embedding, 'cosine')
find_analogy_v2('black', 'education',  'white', word2vec, idx2word, embedding, 'cosine')
find_analogy_v2('man', 'crime',  'woman', word2vec, idx2word, embedding, 'cosine')
find_analogy_v2('man', 'intelligent',  'woman', word2vec, idx2word, embedding, 'cosine')
find_analogy_v2('man', 'intelligent',  'woman', word2vec, idx2word, embedding, 'euclidean')

"""Vectorized implementation is really fast and seems to give more accurate results (probably because loop in find_analogy_v1 uses 
 the distance function defined above whereas find_analogy_v2 uses corresponding function from scikit-learn library"""