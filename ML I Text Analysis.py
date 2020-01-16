#!/usr/bin/env python
# coding: utf-8

# In[7]:


#NLP Methods to Cleanse Text Data and Extract Basic Information
import nltk

# this will be the text document we will analyze
mytext = "We are studying Machine Learning. Our Model learns patterns in data. This learning helps it to predict on new data." 
print("ORIGINAL TEXT = ", mytext)
print('----------------------')

# convert text to lowercase 
mytext = mytext.lower()

# first we will tokenize the text into word tokens 
word_tokens = nltk.word_tokenize(mytext) 
print("WORD TOKENS = ", word_tokens) 
print('----------------------')

# we can also extract sentences if needed 
sentence_tokens = nltk.sent_tokenize(mytext) 
print("SENTENCE TOKENS = ", sentence_tokens) 
print('----------------------')

# lets remove some common stop words
stp_words = ["is","a","our","on",".","!","we","are","this","of","and", "from","to","it","in"]
print("STOP WORDS = ", stp_words)
print('----------------------')

# define cleaned up tokens array 
clean_tokens = []
  
# remove stop words from our word_tokens
for token in word_tokens:
    if token not in stp_words: 
        clean_tokens.append(token)     
print("CLEANED WORD TOKENS = ", clean_tokens)
print('----------------------')

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()

# define cleaned up and lemmatized tokens array 
clean_lemma_tokens = []
clean_stem_tokens = []

# remove stop words from our word_tokens
for token in clean_tokens:
    clean_stem_tokens.append(stemmer.stem(token)) 
    clean_lemma_tokens.append(lemmatizer.lemmatize(token))
    
print("CLEANED STEMMED TOKENS = ", clean_stem_tokens) 
print('----------------------')

print("CLEANED LEMMATIZED TOKENS = ", clean_lemma_tokens) 
print('----------------------')

# get frequency distribution of words 
freq_lemma = nltk.FreqDist(clean_lemma_tokens) 
freq_stem = nltk.FreqDist(clean_stem_tokens)

# import plotting library 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

# set a font size 
chart_fontsize = 30

# plot the frequency chart
plt.figure(figsize=(20,10)) 
plt.tick_params(labelsize=chart_fontsize)
plt.title('Cleaned and Stemmed Words', fontsize=chart_fontsize) 
plt.xlabel('Word Tokens', fontsize=chart_fontsize) 
plt.ylabel('Frequency (Counts)', fontsize=chart_fontsize) 
freq_stem.plot(20, cumulative=False)
plt.show()

# plot the frequency chart
plt.figure(figsize=(20,10)) 
plt.tick_params(labelsize=chart_fontsize)
plt.title('Cleaned and Lemmatized Words', fontsize=chart_fontsize) 
plt.xlabel('Word Tokens', fontsize=chart_fontsize) 
plt.ylabel('Frequency (Counts)', fontsize=chart_fontsize) 
freq_lemma.plot(20, cumulative=False)
plt.show()


# In[11]:


#Parts of Speech Tagging and Named Entity Recognition on Text
# define the sentence that will be analyzed 
mysentence = "Ash is working at Searebral"

print("SENTENCE TO ANALYZE = ", mysentence)
print('----------------------')

# now we will map parts of speech (pos) for the sentence 
word_tk = nltk.word_tokenize(mysentence)
pos_tags = nltk.pos_tag(word_tk)
print("PARTS OF SPEECH FOR SENTENCE = ", pos_tags) 
print('----------------------')

entities = nltk.chunk.ne_chunk(pos_tags) 
print("NAMED ENTITIES FOR SENTENCE = ", entities) 
print('----------------------')


# In[12]:


#create a word vector is using one-hot encoding
#used often to represent categorical data, where each data point belongs to a particular category

#Simple Example of One-Hot Encoded Words

# define the sentence that will be analyzed
mytext = "AI is the new electricity. AI is poised to start a large transformation on many industries."

# we will first tokenize the text 
word_tk = nltk.word_tokenize(mytext) 
words = [w.lower() for w in word_tk]

# create a vocabulary of all relevant words 
vocab = sorted(set(words))
print("VOCABULARY = ", vocab) 
print('----------------------')

# create one hot encoded vectors for each word
for myword in vocab:
    test_1hot = [0]*len(vocab) 
    test_1hot[vocab.index(myword)] = 1
print("ONE HOT VECTOR FOR '%s' = "%myword, test_1hot)


# In[26]:


#Learn Word Embeddings from Text—word2vec
# import the word2vec model
from gensim.models import Word2Vec

# this will be the text document we will analyze
mytext = "AI is the new electricity. AI is poised to start a large transformation on many industries."
print("ORIGINAL TEXT = ", mytext)
print('----------------------')

# convert text to lowercase 
mytext = mytext.lower()

# we can also extract sentences if needed 
sentence_tokens = nltk.sent_tokenize(mytext) 
print("SENTENCE TOKENS = ", sentence_tokens) 
print('----------------------')

# lets remove some common stop words
stp_words = ["is","a","our","on",".","!","we","are","this","of","and", "from","to","it","in"]

# define training data
sentences = []
for sentence in sentence_tokens:
    word_tokens = nltk.word_tokenize(sentence)
    
# define cleaned up tokens array 
    clean_tokens = []

# remove stop words from our word_tokens
    for token in word_tokens:
        if token not in stp_words: 
            clean_tokens.append(token)
    sentences.append(clean_tokens)
        
    print ("TRAINING DATA = ", sentences) 
    print('----------------------')

    # train a new word2vec model on our data - we will use embedding size 20
    word2vec_model = Word2Vec(sentences, size=20, min_count=1)

    # list the vocabulary learned from our corpus 
    words = list(word2vec_model.wv.vocab) 
    print("VOCABULARY OF MODEL = ", words) 
    print('----------------------')

    # show the embeddings vector for some words
    print("EMBEDDINGS VECTOR FOR THE WORD 'ai' = ", word2vec_model["ai"]) 
    print("EMBEDDINGS VECTOR FOR THE WORD 'electricity' = ", word2vec_model["electricity"])


# In[31]:


#Reduce Dimension of Word Embeddings and Plotting the Words
from sklearn.decomposition import PCA

# build training data using word2vec model
training_data = word2vec_model[word2vec_model.wv.vocab]

# use PCA to convert word vectors to 2 dimensional vectors 
pca = PCA(n_components=2)
result = pca.fit_transform(training_data)

# create a scatter plot of the 2 dimensional vectors 
plt.figure(figsize=(20,15)) 
plt.rcParams.update({'font.size': 25}) 
plt.title('Plot of Word embeddings from Text') 
plt.scatter(result[:, 0], result[:, 1], marker="X")

# mark the words on the plot
words = list(word2vec_model.wv.vocab) 
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1])) 
plt.show()


# In[ ]:




