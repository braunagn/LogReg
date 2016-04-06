
# coding: utf-8

# In[ ]:

"""Logistic Regression Model (with One-Hot Encoding)"""

import pandas as pd
import numpy as np
import time
import math
import csv
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from patsy import dmatrices
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scikits.statsmodels.tools import categorical
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer


path1 = "C:\\Users\\lcladmin\\Desktop\\Data Mining\\Datasets\\Recipes\\train.json"
path2 = "C:\\Users\\lcladmin\\Desktop\\Data Mining\\Datasets\\Recipes\\test.json"
recipe = pd.read_json(path1)
test = pd.read_json(path2)
#recipe = recipe.head(100)

start = time.time()

# Do find-and-replace for each ingredient in training data (e.g. "romaine lettuce" to "romaine_lettuce")
sublist_new = []      #empty list for transformed ingredients for single recipe 
ingredients_new = []  #empty list for all transformed ingredients
for sublist in recipe.ingredients:
    for item in sublist:
        item_new = re.sub(" ", "_", item)  #replace spaces with underscores for each ingrdient
        sublist_new.append(item_new)
    temp = " ".join(sublist_new)  #join entire recipe ingredient list into master ingredient list for all training data
    ingredients_new.append(temp)
    sublist_new = []
    
# Do find-and-replace for each ingredient in test data (e.g. "romaine lettuce" to "romaine_lettuce")
sublist_new_test = []      #empty list for transformed ingredients for single recipe 
ingredients_new_test = []  #empty list for all transformed ingredients
for sublist2 in test.ingredients:
    for item2 in sublist2:
        item_new2 = re.sub(" ", "_", item2)
        sublist_new_test.append(item_new2)
    temp2 = " ".join(sublist_new_test)
    ingredients_new_test.append(temp2)
    sublist_new_test = []

#Initialize Python CountVectorizer for Bag-of-Word feature setup and fitting on Training Data
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 6724) #select only top X number of words

train_data_features = vectorizer.fit_transform(ingredients_new)  #initiate bag-of-words function on entire ingredient list

#initialize tf-idf transformer
#tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False).fit(train_data_features)
#train_data_features = tfidf.transform(train_data_features)   #convert train data tf-idf scores



#Initialize Python CountVectorizer (Bag-of-Words) on Test Data so it can be passed to model (.transform ONLY)
test_data_features = vectorizer.transform(ingredients_new_test)  #initiate bag-of-words function on entire ingredient list

#Covert test data to tf-idf scores
#test_data_features = tfidf.transform(test_data_features)



#Create integer values for each cuisine category
cuisine_tags = pd.Series(data=range(0,20), index=["brazilian", "british", "cajun_creole", "chinese", "filipino", "french",
                                                  "greek", "indian", "irish", "italian", "jamaican", "japanese", "korean",
                                                  "mexican", "moroccan", "russian", "southern_us", "spanish", "thai",
                                                  "vietnamese"])
cuisine_cat = recipe["cuisine"].apply(lambda x: cuisine_tags[x])

#Logistic Regression Model
model = LogisticRegression(solver="lbfgs", multi_class="ovr", class_weight="auto")
model = model.fit(train_data_features, cuisine_cat)
predicted = model.predict(test_data_features)

#measure run time of code        
end = time.time()
runtime = end - start
print "Run Time: %.3f secs" % runtime

#generate CSV for submission to Kaggle
cuisine_tags = pd.Series(data=["brazilian", "british", "cajun_creole", "chinese", "filipino", "french",
                                                  "greek", "indian", "irish", "italian", "jamaican", "japanese", "korean",
                                                  "mexican", "moroccan", "russian", "southern_us", "spanish", "thai",
                                                  "vietnamese"])

#format submission file
submission = pd.DataFrame(data={"id":test["id"], "cuisine_id":predicted, "cuisine":np.nan})
submission = submission[["id", "cuisine_id", "cuisine"]]
submission["cuisine"] = submission["cuisine_id"].apply(lambda x: cuisine_tags[x])  #convert cuisine id to cuisine text
submission = submission[["id", "cuisine"]]
submission.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

print submission.head()  #for testing

