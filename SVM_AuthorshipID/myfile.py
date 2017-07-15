#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 00:04:44 2017

@author: ravi
"""
import datetime
import numpy as np
import string
import pandas as pd
from SVM import myfunc
np.random.seed(0123)  # for reproducibility


import csv
authorlist = []
doc_id = []

with open('40auth.csv') as csvfile:
    myreader = csv.reader(csvfile, delimiter=',')
    for row in myreader:
        authorlist.append(int(row[1]))
        doc_id.append(int(row[2]))

#authorlist1 = []
#for i in set(authorlist):
#    authorlist1.append(i)
#
#authorlist = authorlist1

test_output = []
for cnt in range(len(doc_id)):
    temp_id = doc_id[:]
    del temp_id[cnt]
    temp_authorlist = authorlist[:]
    del temp_authorlist[cnt]
    out = myfunc(temp_authorlist , temp_id , authorlist[cnt] , doc_id[cnt])
    test_output.append(out)
    
avg_score = float(sum(test_output))/float(len(test_output))