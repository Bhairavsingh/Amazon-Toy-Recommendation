#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install tkinter


# In[2]:


from tkinter import *
import pandas as pd
import numpy as np


# In[17]:


testdata = pd.read_csv("amazon_co-ecommerce_sample.csv")
test_product_options = np.unique(testdata["product_name"])


# In[22]:


#Function for new window for recomendation results
def open_win_recommend():
    win_recommend = Tk()
    win_recommend.title("Recommended Products")
    
    #Creating frame for result bar
    win_recommend_frame = Frame(win_recommend)
    #Creating scroll bar for result bar
    scroll = Scrollbar(win_recommend_frame)
    scroll.pack(side = RIGHT, fil = Y)
    #Creating window for results
    search_results = Text(win_recommend_frame, width = 50, height = 20, yscrollcommand = scroll.set)
    scroll.config(command = search_results.yview)
    search_results.pack()
    win_recommend_frame.pack()
    win_recommend.mainloop()


#Creating application's main window
win = Tk()
win.title("Amazon Kid's Products")


#Creating top frame for search bar
mainframe = Frame(win)
var = StringVar()
label = Label(mainframe, textvariable = var)
var.set("Product Name")
label.pack()

#Creating entry bar to search product
#entry = Entry(mainframe)
#entry.pack()
#entry.place(anchor = CENTER)

#Creating scroll-down products list
selected_test_product = StringVar(mainframe)
#Setting display for scroll-down
selected_test_product.set("Select Here")
scrolldown_op = OptionMenu(mainframe, selected_test_product, *test_product_options)
scrolldown_op.pack()

#Creating button for recommending algorithm to run
button_Asso = Button(mainframe, text = "Recommend Similar Products", fg = "blue", command = open_win_recommend)
button_Asso.pack()
#Creating button for clustering algorithm to run
button_cluster = Button(mainframe, text = "Perform Clustering", fg = "red")
button_cluster.pack()
mainframe.pack()
#Closing main loop after hitting 'Esc'
win.mainloop()

#Following command can be used for retrieving selected products name.
#selected_test_product.get()


# In[23]:


#Create App icon in Python GUI Application
app = ttk.Tk()
#Application Title
app.title("Kid's Product Recommender")
#Set App icon
app.iconbitmap(r'C:\Users\Bhair\Downloads')
#Calling Main()
app.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




