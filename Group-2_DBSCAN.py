#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from itertools import chain, combinations
import random


# In[2]:


################################# DBSCAN Clustering

import pandas as pd
import math
import numpy as np

#Program to to get clusters with DBSCAN clustering algorithms based on eps and minimum points.
#Input : Data list, Output: Clusters, -1 if observation/ point is noise in data.



def Get_DBSCAN(datalist, eps, minmum_Pts):                      # Function to get clusters from data with DBSCAN
    size = len(datalist)
    cluster_set = [0]*size                                      # Initialize a cluster point list with all 0s
    Cr = 0                                                      # Indicates a point hasn't been considered yet.
    
    

    ##### Check every observation if assigned to clusetr, if not then check eps and min points and accordingly add to the cluster
    for record in range(0, size):
        if not (cluster_set[record] == 0):                      #Initial skip as every value is 0
            continue
        
        neighbor_recordss = Lookforneighbors(datalist, record, eps)  
                                                                # get neighbors of given point 
        if len(neighbor_recordss) < minmum_Pts:                 # Check if neighbors are greater if not, mark point as noise
            cluster_set[record] = -1                        
        else: 
            Cr += 1                                             # Increment cluster number
            #expand_cluster(datalist, cluster_set, record, neighbor_recordss, Cr, eps, minmum_Pts)
            cluster_set[record] = Cr
            i = 0
            while i < len(neighbor_recordss):       
                newrecord = neighbor_recordss[i]
                if cluster_set[newrecord] == -1:                #Already marked -1 then assign it to cluster
                    cluster_set[newrecord] = Cr                     
                elif cluster_set[newrecord] == 0:               #If assigned initially 0 then assign it to cluster
                    cluster_set[newrecord] = Cr
                    newNeighborrecords = Lookforneighbors(datalist, newrecord, eps)
    
                    if len(newNeighborrecords) >= minmum_Pts:       #Again fetch the neighbors for neighbor of the current point.
                        neighbor_recordss = neighbor_recordss + newNeighborrecords
                i += 1              
    
    return cluster_set


def Lookforneighbors(datalist, record, eps):                   #Get neighbors based on given eps value         
    neighbor_recs = []
    size = len(datalist)
    for newrecord in range(0, size):                           #Check each observation in data and get observations within eps provided

        if np.linalg.norm(datalist[record] - datalist[newrecord]) < eps:
            neighbor_recs.append(newrecord)
            
    return neighbor_recs


# In[3]:


final_clustering_df = pd.read_csv("Amazon_Association_final_clustering_df.csv")


# In[4]:


############## Price clustering with DBSCAN

Price_Data=pd.DataFrame(final_clustering_df.Price.unique())
Price_Data.rename(columns = {0:'Price'}, inplace = True) 


DBSCAN_Price=Get_DBSCAN(Price_Data.Price,0.5,10)

DBcluster_Price = pd.DataFrame()
DBcluster_Price['data_index'] = Price_Data.index.values
DBcluster_Price['cluster'] = DBSCAN_Price


# In[5]:



def price_DBclust(cluster_id):
    
    df_indx=DBcluster_Price[DBcluster_Price.cluster == cluster_id].index.values.astype(int)
    df=final_clustering_df.iloc[df_indx,:]
    DBclust1_price=list(df['Names'])
    df_range=list(df['Price'])
    mn=min(df_range)
    mx=max(df_range)
    range_val=str(mn)+'-'+str(mx)
    
    return DBclust1_price, range_val

price_clusters= list()
range_values= list()
for i in range(1,6):
    getclust, range_val=price_DBclust(i)
    price_clusters.append(getclust[0:20])
    range_values.append(range_val)

price_clusters                       ######### Clusters product names list for all clusters
range_values


# In[6]:


############## Rating clustering with DBSCAN

Rating_Data= pd.DataFrame(final_clustering_df['Rating'])

DBSCAN_Rating=Get_DBSCAN(Rating_Data.Rating,0.5,10)

DBcluster_Rating = pd.DataFrame()
DBcluster_Rating['data_index'] = Rating_Data.index.values
DBcluster_Rating['cluster'] = DBSCAN_Rating


# In[ ]:


def rating_DBclust(cluster_id):
    
    df_indx=DBcluster_Rating[DBcluster_Rating.cluster == cluster_id].index.values.astype(int)
    df=final_clustering_df.iloc[df_indx,:]
    DBclust1_rating=list(df['Names'])
    df_range=list(df['Rating'])
    mn=min(df_range)
    mx=max(df_range)
    range_val=str(mn)+'-'+str(mx)
    
    return DBclust1_rating, range_val

rating_clusters= list()
rating_values= list()
for i in range(1,3):
    getclust, range_val=rating_DBclust(i)
    rating_clusters.append(getclust[0:20])
    rating_values.append(range_val)

rating_clusters         ######### Clusters product names list for all clusters
rating_values


# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




