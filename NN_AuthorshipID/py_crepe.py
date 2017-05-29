
# coding: utf-8

# In[4]:

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, LSTM


# In[3]:

def model(dense_outputs, cat_output):
    model = Sequential()
    model.add(Dense(30, input_shape=(10,)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(cat_output, activation='softmax', name='output'))
    
    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    return model


# In[ ]:



