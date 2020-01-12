#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import itertools as it
import random
import time


# In[2]:


#Reading input files
data = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/toy-products-on-amazon/amazon_co-ecommerce_sample.csv")

First_Bought = pd.DataFrame(data["product_name"])
First_Bought = First_Bought.rename(columns = {"product_name" : "First_Bought_Product"})

alsoname_0_to_2500 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Akhil_0_to_2500_alsoname.csv")
alsoname_0_to_2500_pd = pd.DataFrame(alsoname_0_to_2500)
alsoname_0_to_2500_pd = alsoname_0_to_2500_pd.drop(columns = ['6', '7', '8', '9', '10', '11'])

alsoname_2500_to_5000 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Abhishek_2500_to_5000_alsoname.csv")
alsoname_2500_to_5000_pd = pd.DataFrame(alsoname_2500_to_5000)
alsoname_2500_to_5000_pd = alsoname_2500_to_5000_pd.drop(columns = ['6', '7', '8', '9', '10', '11'])

alsoname_5000_to_7500 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Bhairav_5000_to_7500_alsoname.csv")
alsoname_5000_to_7500_pd = pd.DataFrame(alsoname_5000_to_7500)

alsoname_7500_to_end = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Meera_7500_to_end_alsoname.csv")
alsoname_7500_to_end_pd = pd.DataFrame(alsoname_7500_to_end)

alsoname_names = [alsoname_0_to_2500_pd, alsoname_2500_to_5000_pd, alsoname_5000_to_7500_pd, alsoname_7500_to_end_pd]
alsoname = pd.concat(alsoname_names)
alsoname['First_Bought_Product'] = First_Bought
alsoname_cols = alsoname.columns.tolist()
alsoname_cols = alsoname_cols[-1:] + alsoname_cols[:-1]
alsoname = alsoname[alsoname_cols]
alsoname = alsoname.drop(columns = 'Unnamed: 0')
alsoname = alsoname.rename(columns = {"0" : "1", "1" : "2", "2" : "3", "3" : "4", "4" : "5", "5" : "6"})

######################################################################################################################

alsoavgrating_0_to_2500 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Akhil_0_to_2500_alsoavgrating.csv")
alsoavgrating_0_to_2500_pd = pd.DataFrame(alsoavgrating_0_to_2500)
alsoavgrating_0_to_2500_pd = alsoavgrating_0_to_2500_pd.drop(columns = ['6', '7', '8', '9', '10', '11'])

alsoavgrating_2500_to_5000 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Abhishek_2500_to_5000_alsoavgrating.csv")
alsoavgrating_2500_to_5000_pd = pd.DataFrame(alsoavgrating_2500_to_5000)
alsoavgrating_2500_to_5000_pd = alsoavgrating_2500_to_5000_pd.drop(columns = ['6', '7', '8', '9', '10', '11'])

alsoavgrating_5000_to_7500 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Bhairav_5000_to_7500_alsoavgrating.csv")
alsoavgrating_5000_to_7500_pd = pd.DataFrame(alsoavgrating_5000_to_7500)

alsoavgrating_7500_to_end = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Meera_7500_to_end_alsoavgrating.csv")
alsoavgrating_7500_to_end_pd = pd.DataFrame(alsoavgrating_7500_to_end)

alsoavgrating_names = [alsoavgrating_0_to_2500_pd, alsoavgrating_2500_to_5000_pd, alsoavgrating_5000_to_7500_pd, alsoavgrating_7500_to_end_pd]
alsoavgrating = pd.concat(alsoavgrating_names)
alsoavgrating['First_Bought_Product'] = First_Bought
alsoavgrating_cols = alsoavgrating.columns.tolist()
alsoavgrating_cols = alsoavgrating_cols[-1:] + alsoavgrating_cols[:-1]
alsoavgrating = alsoavgrating[alsoavgrating_cols]
alsoavgrating = alsoavgrating.drop(columns = 'Unnamed: 0')
alsoavgrating = alsoavgrating.rename(columns = {"0" : "1", "1" : "2", "2" : "3", "3" : "4", "4" : "5", "5" : "6"})

######################################################################################################################

alsoprice_0_to_2500 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Akhil_0_to_2500_alsoprice.csv")
alsoprice_0_to_2500_pd = pd.DataFrame(alsoprice_0_to_2500)
alsoprice_0_to_2500_pd = alsoprice_0_to_2500_pd.drop(columns = ['6', '7', '8', '9', '10', '11'])

alsoprice_2500_to_5000 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Abhishek_2500_to_5000_alsoprice.csv")
alsoprice_2500_to_5000_pd = pd.DataFrame(alsoprice_2500_to_5000)
alsoprice_2500_to_5000_pd = alsoprice_2500_to_5000_pd.drop(columns = ['6', '7', '8', '9', '10', '11'])

alsoprice_5000_to_7500 = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Bhairav_5000_to_7500_alsoprice.csv")
alsoprice_5000_to_7500_pd = pd.DataFrame(alsoprice_5000_to_7500)

alsoprice_7500_to_end = pd.read_csv("C:/Users/Bhair/OneDrive - University of Oklahoma/Master_of_Science_Data/6th_Semester_Fall_2019/Data_Mining (CS 5593)/Project/Work/Data_Pool/Scrapped Data/Amazon_Association_Data_Meera_7500_to_end_alsoprice_new.csv")
alsoprice_7500_to_end_pd = pd.DataFrame(alsoprice_7500_to_end)

alsoprice_names = [alsoprice_0_to_2500_pd, alsoprice_2500_to_5000_pd, alsoprice_5000_to_7500_pd, alsoprice_7500_to_end_pd]
alsoprice = pd.concat(alsoprice_names)
alsoprice['First_Bought_Product'] = First_Bought
alsoprice_cols = alsoprice.columns.tolist()
alsoprice_cols = alsoprice_cols[-1:] + alsoprice_cols[:-1]
alsoprice = alsoprice[alsoprice_cols]
alsoprice = alsoprice.drop(columns = 'Unnamed: 0')
alsoprice = alsoprice.rename(columns = {"0" : "1", "1" : "2", "2" : "3", "3" : "4", "4" : "5", "5" : "6"})


# In[3]:


#Processing the data to fit the aprori algorithm.
product_names_lists = alsoname.astype(str).values.tolist()
new_pr_names = []
for i in range(len(product_names_lists)):
    new_pr_lists = []
    for j in range(len(product_names_lists[i])):
        if (product_names_lists[i][j] != 'nan'):
            new_pr_lists.append(product_names_lists[i][j])
    new_pr_names.append(sorted(new_pr_lists))


# In[4]:


#Generating combinations of given length from given list.
def combinations_of_sets(sets, number_of_elements):
    comb_sets = list(it.combinations(sets, number_of_elements))
    return comb_sets

#Concatenating dictonaries
def concatenate_dict (first, second): 
    concat = {**first, **second} 
    return concat

#Function for creating list of list if list contains number or tuple.
def Create_List (fre):
    Free = []
    for i in range (len(fre)):
        Few = []
        if (type(fre[i]) != list):
            if (type(fre[i]) == tuple):
                Free.append(list(fre[i]))
            else:
                Few = [fre[i]]
                Free.append(Few)
        else:
            Free.append(fre[i])
    return Free



#Function for generating candidate sets using 'F(k-1) x F(k-1) Method'.
def candidate_set_generation(candidate_set, k):
    #Creating a clone of candidate set.
    candidate_set_2 = candidate_set.copy()
    candidate_set_copying = candidate_set.copy()
    new_candidate_set = []
    index = []
    #Loop for comparing F[k-1] items with F[k-1] items.
    for i in range (len(candidate_set)):
        for j in range (len(candidate_set_2)):
            x = []
            y = []
            if ((candidate_set[i][:(k - 1)] == candidate_set_2[j][:(k - 1)]) and (candidate_set[i][(k - 1)] != candidate_set_2[j][(k - 1)]) and j not in index):
                x = list(candidate_set_copying[i]).copy()
                x.append(candidate_set_2[j][(k - 1)])
                new_candidate_set.append(x)
        index.append(i)
    return new_candidate_set

#Function for counting support count and listing out frequent k_itemsets.
def support_count(in_candidate_set, in_Freq_transaction, k, min_supp):
    inner_Freq_set = []
    inner_support_count = []
    #finding support count by incrementing count after each time similar itemset is found. 
    for i in range (len(in_candidate_set)):
        count = 0
        for j in range (len(in_Freq_transaction)):
            if (tuple(in_candidate_set[i]) in  combinations_of_sets(in_Freq_transaction[j], k)):
                count = count + 1
        #Only if support is more than minimum support, then itemsets are processed further.
        if (count >= min_supp):
            inner_Freq_set.append(in_candidate_set[i])
            inner_support_count.append(count)
    return (inner_Freq_set, inner_support_count)


#Function for generating frequent items.
def frequent_items_gen(Clean_data, support_tr):
    #Creating candidate set of 1_itemset.
    k = 1
    set_1_raw = []
    for i in range (len(Clean_data)):
        set_1_raw.append(combinations_of_sets(Clean_data[i], k))
    set_1 = []
    for i in range (len(set_1_raw)):
        for j in range (len(set_1_raw[i])):
            set_1.append(set_1_raw[i][j])
    #Counting frequency of 1_itemset.
    set_1_val, set_1_raw_count = np.unique(set_1, return_counts = True)

    #Calculating required frequency for given minimum support.
    min_supp = (support_tr/100)*len(Clean_data)
    #Generating frequent 1_itemset.
    Freq_set = []
    All_support_count = []
    #support_count = {}
    for i in range (len(set_1_val)):
        if (set_1_raw_count[i] >= min_supp):
            Freq_set.append(set_1_val[i])
            #support_count[set_1_val[i]] = set_1_raw_count[i]
            All_support_count.append(set_1_raw_count[i])

    #Removing all infrequent trandaction to reduce complexity.
    Freq_transaction = []
    for i in range (len(Clean_data)):
        for j in range (len(Freq_set)):
            if (Freq_set[j] in Clean_data[i]):
                Freq_transaction.append(Clean_data[i])
                break
    #Creating a copy of frequent items for returning with all other frequent items.
    All_Freq_Items = Freq_set.copy()
    while (Freq_set != []):
        k = k + 1
        if (k == 2):
            candidate_set = []
            candidate_set = combinations_of_sets(Freq_set, k)
            Freq_set = []
            support_count_val = []
            #count the frequency and select frequent items
            Freq_set, support_count_val = support_count(candidate_set, Freq_transaction, k, min_supp)
            #Concatenating frequent items from k-1 itemset with k itemset.
            All_Freq_Items = All_Freq_Items + Freq_set
            All_support_count = All_support_count + support_count_val
            #Removing all transaction which contains infrequent intemsets.
            Freq_transaction_copy = Freq_transaction.copy()
            Freq_transaction = []
            for i in range (len(Freq_transaction_copy)):
                for j in range (len(Freq_set)):
                    if (set(Freq_set[j]).issubset(Freq_transaction_copy[i])):
                        Freq_transaction.append(Freq_transaction_copy[i])
                        break
        else:
            candidate_set = []
            candidate_set = candidate_set_generation(Freq_set, k - 1)
            Freq_set = []
            support_count_val = []
            #count the frequency and select frequent items
            Freq_set, support_count_val = support_count(candidate_set, Freq_transaction, k, min_supp)
            #support_count = concatenate_dict(support_count, support_count_add)
            All_Freq_Items = All_Freq_Items + Freq_set
            All_support_count = All_support_count + support_count_val

            Freq_transaction_copy = Freq_transaction.copy()
            Freq_transaction = []
            for i in range (len(Freq_transaction_copy)):
                for j in range (len(Freq_set)):
                    if (set(Freq_set[j]).issubset(Freq_transaction_copy[i])):
                        Freq_transaction.append(Freq_transaction_copy[i])
                        break
    return All_Freq_Items, All_support_count


#Function for generating random samples with given probability.
def random_sample_gen(indata, prob):
    ran_index_samples = random_sample_index_gen(indata, prob)
    out_sample = []
    for i in ran_index_samples:
        out_sample.append(indata[i])
    return out_sample

#Function for generating random index with given probability.
def random_sample_index_gen(indata, prob):
    num = (prob/100)*len(indata)
    sample_length = len(indata)
    gen_num_index = random.sample(range(sample_length), int(num))
    return gen_num_index


#Function for generating association rule.
def Apiori_rules (frequent, supp_cont, min_conf_per):
    New_Freq = Create_List(frequent)
    Rules_antecedent = []
    Rules_consequent = []
    Rules_support_count = []
    Rules_confidence = []
    #For each itemset from frequent itemsets.
    for itemset in New_Freq:
        #Given that, length of itemset is more than 2 items.
        if (len(itemset) >= 2):
            item = itemset[0]
            #Calling rules generator for given itemset and keeping one item to consequent side.
            temp_antecedent, temp_consequent, selected_consequent_support_count, selected_rule_confidence = ap_genrules(itemset, item, supp_cont, min_conf_per, New_Freq)
            Rules_antecedent.append(temp_antecedent)
            Rules_consequent.append(temp_consequent)
            Rules_support_count.append(selected_consequent_support_count)
            Rules_confidence.append(selected_rule_confidence)
    return Rules_antecedent, Rules_consequent, Rules_support_count, Rules_confidence

#Function for association rule cadidates
def ap_genrules(inner_itemset, inner_item, supp_cont, min_conf_per, New_Freq):
    #Minimum confidence as per given by user.
    min_conf = min_conf_per/100
    k = len(inner_itemset)
    #Checking if itemset is a single item.
    if (type(inner_item) != list):
        m = 1
    else:
        m = len(inner_item)
    antecedent = []
    consequent = [inner_item]
    consequent_1st = [inner_item]
    selcted_antecedent = []
    selected_consequent = []
    selected_consequent_support_count = []
    selected_rule_confidence = []
    for i in inner_itemset:
        if (i != inner_item):
            antecedent.append(i)
    for i in range (len(New_Freq)):
        if (inner_itemset == New_Freq[i]):
            Support_count_inner_itemset = supp_cont[i]
        if (consequent == New_Freq[i]):
            Support_count_consequent = supp_cont[i]
    confidence = Support_count_inner_itemset/Support_count_consequent
    #Checking if give itemset with given single consequent has higher confidence than minimum confidence.
    #If yes, then we will find all possible subsets rules of given itemset.
    #If no, then by anti-monotonic property, none of its subset has higher confidence.
    if  confidence >= min_conf:
        selcted_antecedent.append(antecedent)
        selected_consequent.append(consequent)
        selected_consequent_support_count.append(Support_count_consequent)
        selected_rule_confidence.append(confidence)
        list_candidate_consequents = consequent_creator(inner_itemset, consequent_1st)
        list_candidate_antecedents = []
        for e_consequent in list_candidate_consequents:
            list_candidate_antecedents.append(antecedent_creator(inner_itemset, e_consequent))
        for i in range (len(list_candidate_consequents)):
            for j in range (len(New_Freq)):
                if (inner_itemset == New_Freq[j]):
                    Support_count_inner_itemset = supp_cont[j]
                if (list_candidate_consequents[i] == New_Freq[j]):
                    Support_count_consequent = supp_cont[j]
            confidence = Support_count_inner_itemset/Support_count_consequent
            if (confidence >= min_conf):
                selcted_antecedent.append(list_candidate_antecedents[i])
                selected_consequent.append(list_candidate_consequents[i])
                selected_consequent_support_count.append(Support_count_consequent)
                selected_rule_confidence.append(confidence)
    return selcted_antecedent, selected_consequent, selected_consequent_support_count, selected_rule_confidence

#Genrates consequent of given frequent itemset.
def consequent_creator (inner_itemsets, existing_consequent):
    candidate_consequents_mat = [[]]
    candidate_consequents = []
    #Gives all posible combinations of consequents.
    for i in range (len(inner_itemsets)):
        if i > 0:
            each_consequents = [list(j) for j in combinations_of_sets(inner_itemsets, i)]
            if (len(each_consequents) > 0):
                candidate_consequents_mat.append(each_consequents)
    for i in range (1, len(candidate_consequents_mat)):
        for j in range (len(candidate_consequents_mat[i])):
            if (candidate_consequents_mat[i][j] != existing_consequent):
                candidate_consequents.append(candidate_consequents_mat[i][j])
    return candidate_consequents

#Antecedent creator
def antecedent_creator (inner_itemset, x):
    #From given consequent, here we find rest of items which are antecedents.
    itr = inner_itemset.copy()
    for i in x:
        for j in itr:
            if (i == j):
                itr.remove(j)
    return itr

#Finction for listing all rules properly.
def gen_proper_format_rules(x, y):
    inrules_a = []
    inrules_c = []
    for i in range (len(x)):
        for j in range (len(x[i])):
            inrules_a.append(x[i][j])
            inrules_c.append(y[i][j])
    return inrules_a, inrules_c


# In[6]:


#Generating Frequent itemsets with 10% support value.
program_start_time_10 = time.time()
freq_items, supp_coount = frequent_items_gen(new_pr_names, 10)
program_end_time_10 = time.time()

#Generating rules.
antecedent, consequent, support_consequent, confidence = Apiori_rules(freq_items, supp_coount, 1)
#Formating rules.
antecedent_1, consequent_1, = gen_proper_format_rules(antecedent, consequent)
#Generating and formating counts for each items from each rules.
consequent_support, consequent_confidence = gen_proper_format_rules(support_consequent, confidence)


# In[70]:


#Creates the list of consequents.
def apr_pr_list():
    all_prs = []
    for i in range(len(antecedent_1)):
        for j in range(len(antecedent_1[i])):
            all_prs.append(antecedent_1[i][j])
    list_of_prs = np.unique(all_prs)
    return list_of_prs


# In[71]:


#Function used to call apriori algorithm from python django application.
def what_others_bought(index):
    result = []
    count = []
    tot = []
    list_of_prs = apr_pr_list()
    for i in range (len(antecedent_1)):
        if list_of_prs[index] in antecedent_1[i]:
            result.append(consequent_1[i])
            count.append(round(consequent_confidence[i] * consequent_support[i]))
            tot.append(consequent_support[i])
    return result, count, tot


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


antecedent, consequent, support_consequent, confidence = Apiori_rules(freq_items, supp_coount, 1)
antecedent_1, consequent_1, = gen_proper_format_rules(antecedent, consequent)

print('\n', "Rules for overall data with 10% minimum support" , '\n')
for i in range (len(antecedent_1)):
    print(antecedent_1[i], '===>' ,consequent_1[i])


# In[ ]:




