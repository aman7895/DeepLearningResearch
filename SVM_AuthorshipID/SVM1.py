
# coding: utf-8

# In[3]:

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



def myfunc(authors , doc_ids):
    data = pd.read_csv("amannew.csv")
    authorList = data.authorID.unique()
    
    y = data.authorID #classification label
    X = data.drop('authorID', axis = 1) 
    X = data.drop('paraID', axis = 1)
    #X = data.drop('doc_id', axis = 1) #not required for this problem


    # In[7]:

    print(X.dtypes)


    # In[ ]:

    X_train, X_test, y_train, y_test = train_test_split(X, y,
     test_size=0.3,
    random_state=123, stratify = y)


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

    y_pred = clf.predict(X_test)
    print 'R2_score'
    print r2_score(y_test, y_pred)
    print 'Accuracy'
    print clf.score(X_train, y_train)

    print 'Test Accuracy'
    print clf.score(X_test, y_test)



    def preProcessTrainVal(features, labels, groups, K_FOLD = 2):

        # split the data into a training set and a validation set
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()

        print(logo.get_n_splits(features, labels, groups))

    return logo
