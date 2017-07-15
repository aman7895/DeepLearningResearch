
# coding: utf-8

# In[1]:

from __future__ import print_function
from __future__ import division
import json
import py_crepe
import datetime
import numpy as np
import data_helpers
import data
import string
import pandas as pd
from resultMain import myfunc
np.random.seed(0123)  # for reproducibility


# In[2]:
def myfunc1(str):
    import csv
    authorlist = []
    doc_id = []

    #data = pd.read_csv("4auth.csv")

    with open(str) as csvfile:
        myreader = csv.reader(csvfile, delimiter=',')
        for row in myreader:
            authorlist.append(int(row[1]))
            doc_id.append(int(row[2]))


    authorlist1 = []
    for i in set(authorlist):
        authorlist1.append(i)

    authorlist = authorlist1

    test_binary = []
    for doc in doc_id:
        test = [doc]
        out = myfunc(authorlist , test)
        test_binary.append(out)

    s=0

    for line in test_binary:
        s+=sum(line)

    avg_score = float(s)/float(len(test_binary))
    print(avg_score)
    
    return avg_score
    


# In[ ]:



