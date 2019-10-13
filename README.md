Kid’s Entertainment Products Recommendation System Using Association Rule Mining for Amazon Data.

2.Authors:
Abhishek Gupta          : abhigupta@ou.edu
Akhil Potdar            : akhilpotdar@ou.edu
Bhairavsingh Ghorpade   : Bhairavsingh.Ghorpade@ou.edu
Meera Bankar            : Meerabankar@ou.edu

3. Objectives of the project: To build a recommendation system for kid’s toy and book products for Amazon using association rule mining and clustering techniques.
-Recommend product based on association rule mining (The following items are bought together) and clustering (if you bought this...you might want this, similar products..)
-Recommend products using clustering over below parameters
● Price of the product, Rating of Product, Type of product, Brand of product

4. Significance
4.1 Application:
Recommender systems are designed with ideas from data mining, machine learning, and user profiling research in order to provide end-users with more proactive and personalized information retrieval applications. Two popular approaches have come to dominate. First, content-based techniques leverage the availability of rich item descriptions to identify new items that are similar to those that a user has liked in the past. In contrast, collaborative
recommendation tries to find some users who share similar tastes with the given user and recommends articles they like to that user. Recommendation systems are the most important part of an eCommerce business as it decides the growth of business. Examples of such applications include recommending books, CDs, and other products at Amazon.com, movies by MovieLens, Myntra, Flipkart and news at VERSIFI Technologies (formerly AdaptiveInfo.com). Moreover, some of the vendors have incorporated recommendation capabilities into their commerce servers. Today, customers are preferring a more faster way of purchase, from the essential goods to luxury items every user tries to buy from the digital platform. Hence the demand for effective recommendation systems is increasing every day.

4.2 Datasets and Sources
For this project, we are using ‘Toy products on amazon’ dataset. This dataset is accessed from Kaggle open-source data repository. Dataset has 10000 observations. Each observation represents one user. Dataset has 17 attributes. The original dataset has the following attributes. The meaning, data type, size per attribute is given in the following table. Attributes Meaning Type Val size Size/Attribute
uniq_id
Buyer unique Id.
String
32
320000
product_name
Name of products.
String
24
240000
manufacturer
Product Manufacturer
String
34
340000
price
Product Price
String
6
60000
number_available_in_stock
Stock available
String
6
60000
number_of_reviews
Number of reviews
Integer
4
40000
number_of_answered_questions
Number of answered questions
Integer
4
40000
average_review_rating
Average ratings.
String
6
60000
amazon_category_and_sub_category
Category and subcategories of product
String
24
240000
customers_who_bought_this_item_also_bought
Items bought by customer after buying the given product.
String
34
340000
description
Description of product.
String
48
480000
product_information
Product information.
String
40
400000
product_description
Description of product.
String
48
480000
items_customers_buy_after_viewing_this_item
List of items that customer bought after viewing the given product.
String
32
320000
customer_questions_and_answer
Customer question and answer pairs.
String
48
480000
customer_reviews
Product's customer reviews.
String
24
240000
sellers
Seller information of the product.
String
16
160000

Record size: 430 Bytes (This is avg record size, it may differ per record as every string has different length and we haven’t defined max length for attributes as it is not required in python), Dataset size: 34MB, Also few data attributes need to be converted to other data types. Data source: https://www.kaggle.com/PromptCloudHQ/toy-products-on-amazon

4.3 Data Mining Tasks and their Purpose
The application will look into details of all the products and the attributes associated with it. Since our aim is to work with all the products in the two categories of toys and books, considering the volume of the dataset, we intend to implement association mining and clustering techniques to achieve our objective. The clustering of products will give help us to group toys or books based on similarities in terms of different attributes like price or ratings. So, if the customer bought one product belonging to one category, he/she might be interested in other products belonging to the same clusters which will be recommended by the application. The association mining will give us the association of one product with others which may or may not belong to the same clusters.

4.4 Type of Data Mining algorithm needed
In a process of grouping data into clusters, we will be using different clustering techniques optimizing the intra-class similarity and finding its minimum [Rana,2014]. We will use clustering to group similar toys and recommend them to the user in the application. For clustering, we will use multiple algorithms, each of which can vary in its method of computation. For instance, k-means, density based, hierarchical, DBSCAN are some of the algorithms which can be implemented to carry out the task of clustering. For recommendation of similar products bought
together, we will be using association mining. Association rule mining well help us identify frequent patterns, associations, correlations, or causal structures among different attributes of each product of the dataset (Kaur, 2013). We will be using different algorithms including Apriori, FP tree frequency, partitioning based algorithms (Xinxiang, 2017) to compare their relative performance and efficiency.
Finally, we will also develop an application giving recommendation system that uses both association mining with clustering techniques (Semesters, 2013 ; Lian, 2018; Swami 1997). implementing better, faster and efficient recommendation system for Kid’s toy and books.

5. References:
1) Charanjeet Kaur (2013) .Association Rule Mining using Apriori Algorithm: A Survey.International Journal of Advanced Research in Computer Engineering & Technology (IJARCET) Volume 2, Issue 6, June 2013.
2) Lian, S., Gao, J., & Li, H. (2018). A Method of Mining Association Rules for Geographical Points of Interest. ISPRS International Journal of Geo-Information, 7(4), 146. doi: 10.3390/ijgi7040146
3) Rana, Goldy, and Silky Azad (2014). Analysis of Clustering Algorithms in E-Commerce Using WEKA - Semantic Scholar. International Journal of Computer Science & Management Studies.
4) Smetsers, Rick (2013). Association rule mining for recommender systems. Master Track Human Aspects of Information Technology, at the School of humanities of Tilburg University.
5) Yutang Liu, Qin Zhang (2017). Research on Association Rules Mining Algorithm Based on Large Data. Revista de la Facultad de Ingeniería U.C.V., Vol. 32.
