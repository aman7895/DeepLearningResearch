
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
np.random.seed(0123)  # for reproducibility


# In[2]:

# set parameters:
def myfunc(authorlist , doc_id, tdoc):
    subset = None

    #Whether to save model parameters
    save = False
    model_name_path = 'params/crepe_model.json'
    model_weights_path = 'params/crepe_model_weights.h5'

    #Maximum length. Longer gets chopped. Shorter gets padded.
    maxlen = 1014

    #Model params
    #Filters for conv layers
    nb_filter = 128 #initially 256
    #Number of units in the dense layer
    dense_outputs = 512 #Initially 1024
    #Conv layer kernel size
    filter_kernels = [3, 3, 3, 3, 3, 3]
    #Number of units in the final output layer. Number of classes.

    #Compile/fit params
    batch_size = 32
    nb_epoch = 10

    print('Loading data...')
    #Expect x to be a list of sentences. Y to be a one-hot encoding of the
    #categories.

    ### 515-1122-122 and 1573 with remove 6 layers
    #authorlist=[121, 479 , 649 ]
    #doc_id = 14706

    #authorlist=[114, 1492, 124, 1228]
    #doc_id = [206, 205]
    cat_output = len(authorlist) #binary in the last layer

    # def main(authorlist, doc_id):
    

    ((trainX, trainY), (valX, valY)) = data_helpers.load_ag_data(authors = authorlist, docID = doc_id, tdoc = tdoc)

    print('Creating vocab...')
    vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()


    #trainX = data_helpers.encode_data(trainX, maxlen, vocab, vocab_size, check)
    #test_data = data_helpers.encode_data(valX, maxlen, vocab, vocab_size, check)

    print('Build model...')

    classes = len(authorlist)
    (model, sgd, model_weights_path) = py_crepe.build_model(classes, filter_kernels,
                                                            dense_outputs, maxlen, vocab_size, nb_filter)

    vocab_size

    print('Fit model...')
    initial = datetime.datetime.now()
    for e in xrange(nb_epoch):
        xi, yi = data_helpers.shuffle_matrix(trainX, trainY)
        xi_test, yi_test = data_helpers.shuffle_matrix(valX, valY)
        if subset:
            batches = data_helpers.mini_batch_generator(xi[:subset], yi[:subset],
                                                        vocab, vocab_size, check,
                                                        maxlen,
                                                        batch_size=batch_size)
        else:
            batches = data_helpers.mini_batch_generator(xi, yi, vocab, vocab_size,
                                                        check, maxlen,
                                                        batch_size=batch_size)

        test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                         vocab_size, check, maxlen,
                                                         batch_size=batch_size)

        accuracy = 0.0
        loss = 0.0
        step = 1
        start = datetime.datetime.now()
        print('Epoch: {}'.format(e))
        for x_train, y_train in batches:

            f = model.train_on_batch(x_train, y_train)
            loss += f[0]
            loss_avg = loss / step
            accuracy += f[1]
            accuracy_avg = accuracy / step
            if step % 100 == 0:
                print('  Step: {}'.format(step))
                print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
            step += 1

        test_accuracy = 0.0
        test_loss = 0.0
        test_step = 1

        for x_test_batch, y_test_batch in test_batches:
            f_ev = model.test_on_batch(x_test_batch, y_test_batch)
            test_loss += f_ev[0]
            test_loss_avg = test_loss / test_step
            test_accuracy += f_ev[1]
            test_accuracy_avg = test_accuracy / test_step
            test_step += 1
        stop = datetime.datetime.now()
        e_elap = stop - start
        t_elap = stop - initial
        print('Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_accuracy_avg, e_elap, t_elap))

    if save:
        print('Saving model params...')
        json_string = model.to_json()
        with open(model_name_path, 'w') as f:
            json.dump(json_string, f)

    model.save_weights(model_weights_path)

    import cPickle as pickle
    with open('sgd.pickle', 'wb') as handle:
        pickle.dump(sgd, handle, protocol=pickle.HIGHEST_PROTOCOL)



    del trainX, trainY, valX, valY

    model.load_weights(model_weights_path)

    #from keras.optimizers import SGD
    #sgd = SGD(lr=0.01, momentum=0.9, nesterov= True)

    # Compile model again (required to make predictions)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])



    def predictModel(model, testX):
        # Function to take input of data and return prediction model
        predY = np.array(model.predict(testX))

        predYList = predY[:]
        entro = []

        flag = False
        import math
        for row in predY:
            entroval = 0
            for i in row:
                if(i <= 0):
                    flag = True
                    pass
                else:
                    entroval += (i * (math.log(i , 2)))
            entroval = -1 * entroval
            entro.append(entroval)

        if(flag == False):
            yx = zip(entro, predY)
            yx = sorted(yx, key = lambda t: t[0])
            newPredY = [x for y, x in yx]
            predYEntroList = newPredY[:int(len(newPredY)*0.3)] # Reduce this 
            predY = np.mean(predYEntroList, axis=0)
        else:
            predY = np.mean(predYList, axis=0)

        return (predYList, predY)

    test_binary = [] 
    #d = []
    #d.append(doc_id)
    #for docs in d:
    (testX, testY) = data_helpers.load_doc_data(authors = authorlist, docID = doc_id)
    testX = data_helpers.encode_data(testX, maxlen, vocab, vocab_size, check)
    (predYList, predY) = predictModel(model, testX)
    testY = np.array(testY)
    testY = testY.mean(axis = 0)
    predLocation = predY.tolist().index(max(predY))
    if predLocation == testY:
        test_binary.append(1)
    else:
        test_binary.append(0)


    from IPython.display import clear_output
    clear_output()




    #predY = np.array(model.predict(testX, batch_size=batch_size))

    #predY


    #predY.mean(axis = 0)

    return test_binary

    """
    feature_model = py_crepe.build_feature_model()
    feature_trainX = feature_model.predict(trainX)
    feature_testX = feature_model.predict(testX)
    """


# In[ ]:



