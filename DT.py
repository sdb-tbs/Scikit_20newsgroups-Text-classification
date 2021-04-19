
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 06:57:28 2020

@author: Soodabeh
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time
 
categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
 
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42) 

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

""" Decision Tree
""" 
text_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),('clf', DecisionTreeClassifier(random_state=42))])
text_clf.fit(twenty_train.data, twenty_train.target)
#Pipeline(...)
twenty_test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle=True, random_state=42)
docs_test= twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted==twenty_test.target))
# Gridsearch

#------>min sample splite is internal node while min sample leaf is an external node
#random state is not a parameter to tune (https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680)
#max_feature : 
param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
              'clf__min_samples_split': range(1,10,1), 'clf__max_depth': range(2,100,5),'clf__min_samples_leaf': range(1,100,10)}
grid_search_cv = GridSearchCV(text_clf, param_grid, verbose=1,  cv=3,n_jobs=-1)

gs = grid_search_cv.fit(twenty_train.data, twenty_train.target)
print("Best parameter (CV score=%0.3f):" % grid_search_cv.best_score_)
print(grid_search_cv.best_params_)








"""Fit and Test with best parameters
"""

start = time.time()

twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42) 

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

"""# Decision Tree
""" 
text_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),('clf',  DecisionTreeClassifier(random_state=42, max_depth = 100,  min_samples_split = 2, min_samples_leaf = 1,  ) )])
text_clf.fit(twenty_train.data, twenty_train.target)
#Pipeline(...)
twenty_test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle=True, random_state=42)
docs_test= twenty_test.data
predicted = text_clf.predict(docs_test)
stop = time.time()
print(np.mean(predicted==twenty_test.target))
print('execution time is'+str(stop - start))
