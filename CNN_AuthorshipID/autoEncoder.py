
# coding: utf-8

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D

def auto_model(maxlen, vocab_size):
    #Define what the input shape looks like
    
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(maxlen, vocab_size)))
    #model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(vocab_size, activation='sigmoid'))
    
    model.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])
    
    filepath = 'params/auto_model_weights.h5'
    
    return (model, filepath)

def fcc_auto_model(maxlen, vocab_size, filepath, dense_outputs, classes):
    #Define what the input shape looks like
    
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(maxlen, vocab_size), trainable = False))
    model.add(Dense(64, activation='relu', trainable = False))
    model.add(Dropout(0.3, trainable = False))
    model.add(Dense(32, activation='relu', trainable = False))

    model.add(Dense(64, activation='relu', trainable = False))
    model.add(Dense(128, activation='relu', trainable = False))
    model.add(Dense(vocab_size, activation='sigmoid', trainable = False))
    
    model.load_weights(filepath)
    
    model.pop()  # Dense(vocab_size, activation='sigmoid')(decoded)
    
    model.pop()  # Dense(128, activation='relu')(decoded)
    
    model.pop()  # Dense(64, activation='relu')(encoded)
    
    model.add(Flatten())
    
    model.add(Dense(32, activation='relu', name = 'dense1'))
    
    model.add(Dropout(0.5, name = 'dropout1'))
    
    model.add(Dense(classes, activation='softmax', name='output'))
    
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

    return (model, filepath)
