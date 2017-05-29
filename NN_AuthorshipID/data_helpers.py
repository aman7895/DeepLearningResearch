
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


# In[2]:

# This function is called from Main and expects train and test values for x and y
def load_ag_data(authors = None, docID = None): 
    
    data = pd.read_csv("amannew.csv")
    
    print(data.shape)
    
    labels = []
    groups = []
    features = []
    size = []
    
    authorList = authors
    
    for auth in authorList:
        current = data.loc[data['author_id'] == auth]
        size.append(current.shape[0])

    print("Min: %s" % (min(size)))
    print("Max: %s" % (max(size)))
    
    data = data.loc[data['doc_id'] != docID]
    
    print data.shape
    
    for auth in authorList:
        current = data.loc[data['author_id'] == auth]

        # current = current.sample(n = samples)
        feat = current[["f1", "f2", "f3", "f4", 
                        "f5", "f6", "f7", "f8",
                        "f9", "f10"]].values.tolist()
        features = features + feat
        
        labels = labels + [authorList.index(author_id) for author_id in current.author_id.tolist()]
        
        doc = current["doc_id"].tolist()
        groups = groups + doc
        
    labels = to_categorical(np.asarray(labels))
    
    features = np.array(features)
    labels = np.array(labels)
    groups = np.array(groups)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=123)
    
    # return (texts, labels, labels_index, samples)

    return ((X_train, y_train), (X_test, y_test), authorList)
