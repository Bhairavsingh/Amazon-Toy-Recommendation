
# coding: utf-8

# In[27]:


pip install selenium


# In[28]:


from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import math
import numpy as np


# In[29]:


#Reading input files
data = pd.read_csv("amazon_co-ecommerce_sample .csv")


# In[31]:


#Chrome driver for web scraping
driver = webdriver.Chrome("C:/Users/meera/Downloads/chromedriver_win32 (1)/chromedriver")


# In[32]:


#Function for detecting nan values from list of strings.
def is_nan_in_string(datalist):
    for i in range (len(datalist)):
        if (str(datalist[i]) == "nan"):
            datalist[i] = "Nothing"
    return datalist

#Function for converting list of charecters to a string.
def convert_string(x): 
    # initialization of string to "" 
    new = "" 
    # traverse in the string  
    for i in x:
        new += i  
    # return string  
    return new 

#Function for converting links from charecters to link type structure.
def get_link_one_by_one(indata):
    lot = []
    #Storing a stop sign.
    for i in indata:
        lot.append(i)
    lot.append('!')
    i = 0
    all_links = []
    #Accessing all charecters from input.
    while i < (len(lot)):
        link = []
        #Until a space is found.
        while lot[i] != " ":
            if (i == len(lot) - 1):
                break;
            link.append(lot[i])
            i += 1
        #Based on structure of input, +3 position are puched forward.
        i += 3
        all_links.append(convert_string(link))
    return all_links

#Function for links to be in the form of list of list per product.
def get_links(indata):
    all_links = []
    for i in range (len(indata)):
        all_link = get_link_one_by_one(indata[i])
        all_links.append(all_link)
    return all_links

#FUnction for web scaping.
def get_info(link):
    if (link != 'Nothing' and link != ''):
        driver.get(link)
        page_content = driver.page_source
        pulled_content = BeautifulSoup(page_content)
        toy_name = pulled_content.find('span', attrs={'class':'a-size-large'})
        #average_rating = pulled_content.find('span', attrs={'class':'a-icon-alt'})
        average_rating = pulled_content.find('span', attrs={'data-hook':'rating-out-of-text'})
        #manufacturer = pulled_content.find('span', attrs={'class':'a-link-normal'})
        price = pulled_content.find('span', attrs={'class':'a-color-price'})
    
        name_string_whole = str(toy_name)
        toy_name_string = name_string_whole[45 : (len(name_string_whole) - 7)]
        toy_name_string = toy_name_string.strip()
    
        avg_string = str(average_rating)
        average_rating_string = avg_string[72:75]
        #average_rating_string = avg_string[81:83]
        #average_rating_string = avg_string[25:28]
        
        price_string_whole = str(price)
        price_string = price_string_whole[28 : (len(price_string_whole) - 7)]
        price_name_string = price_string.strip()
        
        return toy_name_string, average_rating_string, price_name_string
    else:
        toy_name_string = 'Nothing'
        average_rating_string = 0
        price_name_string = ''
        return toy_name_string, average_rating_string, price_name_string

#Function for getting all required data (Product name and average rating) for entire column of links.
def give_list_of_data(list_of_links):
    #indata = is_nan_in_string(indata_all)
    #list_of_links = get_links(indata)
    toy_name_entire_list = []
    average_rating_entire_list = []
    price_entire_list = []
    for i in range (len(list_of_links)):
        toy_name_entire_list_for_row = []
        average_rating_entire_list_for_row = []
        price_entire_list_for_row = []
        for j in range (len(list_of_links[i])):
            toyname, avgrating, price = get_info(list_of_links[i][j])
            toy_name_entire_list_for_row.append(toyname)
            average_rating_entire_list_for_row.append(avgrating)
            price_entire_list_for_row.append(price)
        toy_name_entire_list.append(toy_name_entire_list_for_row)
        average_rating_entire_list.append(average_rating_entire_list_for_row)
        price_entire_list.append(price_entire_list_for_row)
    return toy_name_entire_list, average_rating_entire_list, price_entire_list


#Get all links. (Just for trail)
def screpgetgo(idata):
    x = is_nan_in_string(idata)
    y = get_links(x)
    N_product_name, N_Avaerage_rating, N_Price = give_list_of_data(y)
    return N_product_name, N_Avaerage_rating, N_Price


# In[33]:


#Converting charecters to links.
strings_links = is_nan_in_string(data["customers_who_bought_this_item_also_bought"])
#Getting list of links.
links = get_links(strings_links)


# In[34]:


#Getting information from amazon website.


######################################################################################################################
#Remove comment sign for your code below
######################################################################################################################

#For Akhil
#also_name, also_avg_rating, also_price = give_list_of_data(links[:2500])

#For Abhishek
#also_name, also_avg_rating, also_price = give_list_of_data(links[2500:5000])

#For Bhairav
#also_name, also_avg_rating, also_price = give_list_of_data(links[5000:7500])

#For Meera
also_name, also_avg_rating, also_price = give_list_of_data(links[7500:])


# In[137]:


pd_also_name = pd.DataFrame(also_name)
pd_also_avg_rating = pd.DataFrame(also_avg_rating)
pd_also_price = pd.DataFrame(also_price)


######################################################################################################################
#Remove comment sign for your code below
######################################################################################################################


#pd_also_name.to_csv('Amazon_Association_Data_Akhil_0_to_2500_alsoname.csv')
#pd_also_avg_rating.to_csv('Amazon_Association_Data_Akhil_0_to_2500_alsoavgrating.csv')
#pd_also_price.to_csv('Amazon_Association_Data_Akhil_0_to_2500_alsoprice.csv')

######################################################################################################################

#pd_also_name.to_csv('Amazon_Association_Data_Abhishek_2500_to_5000_alsoname.csv')
#pd_also_avg_rating.to_csv('Amazon_Association_Data_Abhishek_2500_to_5000_alsoavgrating.csv')
#pd_also_price.to_csv('Amazon_Association_Data_Abhishek_2500_to_5000_alsoprice.csv')

######################################################################################################################

#pd_also_name.to_csv('Amazon_Association_Data_Bhairav_5000_to_7500_alsoname.csv')
#pd_also_avg_rating.to_csv('Amazon_Association_Data_Bhairav_5000_to_7500_alsoavgrating.csv')
#pd_also_price.to_csv('Amazon_Association_Data_Bhairav_5000_to_7500_alsoprice.csv')

######################################################################################################################


# In[ ]:


pd_also_name.to_csv('Amazon_Association_Data_Meera_7500_to_end_alsoname.csv')
pd_also_avg_rating.to_csv('Amazon_Association_Data_Meera_7500_to_end_alsoavgrating.csv')
pd_also_price.to_csv('Amazon_Association_Data_Meera_7500_to_end_alsoprice.csv')


# In[142]:


flat_price=pd_also_price.copy()


# In[145]:


type(flat_avg_rating[0][0])


# In[143]:



for i in range(0,6):
    for j in range(0,len(flat_price[i])):
        if (str(flat_price[i][j]).find('£')>=0):
            start=str(flat_price[i][j]).find('£')+1
            end=len(str(flat_price[i][j]))
            flat_price[i][j]=str(flat_price[i][j][start:end])
        else:
            flat_price[i][j]=''
flat_price


# In[141]:


flat_price.to_csv('Amazon_Association_Data_Meera_7500_to_end_alsoprice_new.csv')

