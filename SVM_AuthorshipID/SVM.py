#te
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.externals import joblib
from sklearn.svm import SVC


# In[2]:
def myfunc(authors , doc_ids , test_author , test_doc):
    
#    authors = temp_authorlist
#    doc_ids = temp_id
#    test_author = authorlist[0]
#    test_doc = doc_id[0]    
    df = pd.read_csv("amannew.csv")
    data = pd.DataFrame(data=None, columns=df.columns)
    for i,j in zip(authors , doc_ids): 
        temp = df.loc[(df['authorID'] == i) & (df['docID'] == j)]
        data = data.append(temp, ignore_index=True)              
    
    
       
    # In[6]:
    
    y_train = data.authorID #classification label
    X_train = data.drop('authorID', axis = 1) 
    #X = data.drop('doc_id', axis = 1) #not required for this problem
    
    
    # In[7]:
    
    #    print(X.dtypes)
    
    
    # In[ ]:
    #works
    y_test = test_author
    X_test = df.loc[(df['authorID'] == test_author) & (df['docID'] == test_doc)]
    X_test = X_test.drop('authorID', axis = 1)                 
    
    # In[ ]:
    
    X_train_scaled = preprocessing.scale(X_train)
    print X_train_scaled
    
    
    # In[ ]:
    
    print X_train_scaled.mean(axis=0)
    
    
    # In[ ]:
    
    print X_train_scaled.std(axis=0)
    
    
    # In[ ]:
    
    # Pre-processing 
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    print X_train_scaled.mean(axis=0)
    # [ 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    print X_train_scaled.std(axis=0)
    # [ 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    
    
    # In[ ]:
    
    X_test_scaled = scaler.transform(X_test)
    
    print X_test_scaled.mean(axis=0)
    
    print X_test_scaled.std(axis=0)
    
    
    # In[ ]:
    
    ## SVC
    clf = SVC(kernel='linear', verbose= True, C= 1)
    clf.fit(X_train,y_train)
    
    
    # In[ ]:
    
    #    y_pred = clf.predict(X_test)
    #    print 'R2_score'
    #    print r2_score(y_test, y_pred)
    #    print 'Accuracy'
    #    print clf.score(X_train, y_train)
    
    #    print 'Test Accuracy'
    return clf.score(X_test, y_test)
    
    
