#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from smart_open import smart_open
# nltk.download('stopwords')  # run once
from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
from gensim import corpora
from pprint import pprint
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
# from jupyterthemes import jtplot
# jtplot.style(theme='monokai', context='notebook',
#              ticks=True, grid=False)
import math
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot
import operator
import pandas as pd
# jtplot.style(theme='monokai', context='notebook',
#              ticks=True, grid=False)


# In[ ]:


df= pd.read_csv('C:/Users/Bhair/final_clustering_df.csv')
X = np.asarray(df.iloc[:,[1,2]])

#Function for calculating distances.
def euclidean_distance_word(dat, rand_medians,word_sim):
    dist_list = []
    for i in range(len(rand_medians)):
        summation = np.sum((X-rand_medians[i])**2, axis=1)
        summation = normalize(summation.reshape(-1,1))
        summation = summation + (1-word_sim).reshape(-1,1)
#         summation = word_sim
        dist_list.append(np.sqrt(summation))
    dist_list = np.asarray(dist_list)
    return (dist_list)

#Function for calculating similiarity between products.
def word_sim_calc(df,item):
    file_docs = []
    for i in range(len(df)):
        tokens = sent_tokenize(df['Names'][i])
        file_docs.append(tokens)
    mydict = corpora.Dictionary([simple_preprocess((" ".join(line))) for line in file_docs])
    corpus = [mydict.doc2bow(simple_preprocess((" ".join(line)))) for line in file_docs]
    tfidf = gensim.models.TfidfModel(corpus)
    #workdir = r'C:\Users\Bhair\ '
    workdir = r'C:\Users\Bhair\Data_Mining_Application\DM_Web_App\ '
    sims = gensim.similarities.Similarity(workdir,tfidf[corpus],num_features=len(mydict))
    query_doc_tf_idf = tfidf[corpus[item]]
    word_sim = sims[query_doc_tf_idf]
    return (word_sim)
#Function to access from python django.
def similar_prod(item,no_item):
    data = df
    user = []
    X = np.asarray(data.iloc[:,[1,2]])
    user.append(X[item])
    wordsim = word_sim_calc(data,item)
    dist_array = euclidean_distance_word(X,user,wordsim)
    dict = { i : dist_array[:,i] for i in range(0, dist_array.shape[1] ) }
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1))
    similar_prod_index = sorted_dict[0:(no_item+1)]
    result = df.iloc[np.asarray(similar_prod_index)[:,0]]
    name = result['Names'].tolist()
    price = result['Price'].tolist()
    rating = result['Rating'].tolist()
    return name,price,rating


# In[3]:





# In[4]:





# In[ ]:





# In[ ]:




