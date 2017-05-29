# coding: utf-8
import pandas as pd

def getCharAuthorData(authors = None, doc = None, documentTable = 'document_unicode', chunk_size = 1000):
    df = pd.read_csv("aman_ml_authors_20.csv")
    print(df.dtypes)
    print("Data Frame created: Shape: %s" % (str(df.shape)))
    return df

