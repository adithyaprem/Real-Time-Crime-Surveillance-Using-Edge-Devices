import make_hof_half
from make_hof_half import get_processed_hof
import pandas as pd
import os,datetime
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pathlib
import pickle as pkl

from video_stuff import get_frames_per
from video_stuff import hof,getFlow
from time import time 
# from make_emi_data import emirnn_preprocess

# EMI imports
from rnn import EMI_DataPipeline
from rnn import EMI_BasicLSTM, EMI_FastGRNN, EMI_FastRNN, EMI_GRU
from emirnnTrainer import EMI_Trainer, EMI_Driver
import utils
import os 
import numpy as np
import tensorflow as tf

from metrics import emi_model_metrics
import numpy as np
import matplotlib

import os,datetime
from time import time

# from emirnn_preprocess import emirnn_preprocess
def lstm_experiment_generator(params, data, path = './DSAAR/64_16/'):
    """
        Function that will generate the experiments to be run.
        Inputs : 
        (1) Dictionary params, to set the network parameters.
        (2) Name of the Model to be run from [EMI-LSTM, EMI-FastGRNN, EMI-GRU]
        (3) Path to the dataset, where the csv files are present.
    """
    
    #Copy the contents of the params dictionary.
    lstm_dict = {**params}
    
    #---------------------------PARAM SETTING----------------------#
    
    # Network parameters for our LSTM + FC Layer
    NUM_HIDDEN = params["NUM_HIDDEN"]
    NUM_TIMESTEPS = params["NUM_TIMESTEPS"]
    ORIGINAL_NUM_TIMESTEPS = params["ORIGINAL_NUM_TIMESTEPS"]
    NUM_FEATS = params["NUM_FEATS"]
    FORGET_BIAS = params["FORGET_BIAS"]
    NUM_OUTPUT = params["NUM_OUTPUT"]
    USE_DROPOUT = True if (params["USE_DROPOUT"] == 1) else False
    KEEP_PROB = params["KEEP_PROB"]

    # For dataset API
    PREFETCH_NUM = params["PREFETCH_NUM"]
    BATCH_SIZE = params["BATCH_SIZE"]

    # Number of epochs in *one iteration*
    NUM_EPOCHS = params["NUM_EPOCHS"]
    # Number of iterations in *one round*. After each iteration,
    # the model is dumped to disk. At the end of the current
    # round, the best model among all the dumped models in the
    # current round is picked up..
    NUM_ITER = params["NUM_ITER"]
    # A round consists of multiple training iterations and a belief
    # update step using the best model from all of these iterations
    NUM_ROUNDS = params["NUM_ROUNDS"]
    LEARNING_RATE = params["LEARNING_RATE"]

    # A staging direcory to store models
    MODEL_PREFIX = params["MODEL_PREFIX"]
    
    #----------------------END OF PARAM SETTING----------------------#
    
    #----------------------DATA LOADING------------------------------#
    
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    x_val, y_val = data['x_val'], data['y_val']
    

    # BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
    # step of EMI/MI RNN
    BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)
    BAG_TRAIN = np.argmax(y_train[:, 0, :], axis=1)
    BAG_VAL = np.argmax(y_val[:, 0, :], axis=1)
    NUM_SUBINSTANCE = x_train.shape[1]
    print("x_train shape is:", x_train.shape)
    print("y_train shape is:", y_train.shape)
    print("x_test shape is:", x_val.shape)
    print("y_test shape is:", y_val.shape)
    
    #----------------------END OF DATA LOADING------------------------------#    
    
    #----------------------COMPUTATION GRAPH--------------------------------#
    
    # Define the linear secondary classifier
    def createExtendedGraph(self, baseOutput, *args, **kwargs):
        W1 = tf.Variable(np.random.normal(size=[NUM_HIDDEN, NUM_OUTPUT]).astype('float32'), name='W1')
        B1 = tf.Variable(np.random.normal(size=[NUM_OUTPUT]).astype('float32'), name='B1')
        y_cap = tf.add(tf.tensordot(baseOutput, W1, axes=1), B1, name='y_cap_tata')
        self.output = y_cap
        self.graphCreated = True

    def restoreExtendedGraph(self, graph, *args, **kwargs):
        y_cap = graph.get_tensor_by_name('y_cap_tata:0')
        self.output = y_cap
        self.graphCreated = True

    def feedDictFunc(self, keep_prob=None, inference=False, **kwargs):
        if inference is False:
            feedDict = {self._emiGraph.keep_prob: keep_prob}
        else:
            feedDict = {self._emiGraph.keep_prob: 1.0}
        return feedDict

    EMI_BasicLSTM._createExtendedGraph = createExtendedGraph
    EMI_BasicLSTM._restoreExtendedGraph = restoreExtendedGraph

    if USE_DROPOUT is True:
        EMI_Driver.feedDictFunc = feedDictFunc
    
    inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
    emiLSTM = EMI_BasicLSTM(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS,
                            forgetBias=FORGET_BIAS, useDropout=USE_DROPOUT)
    emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy',
                             stepSize=LEARNING_RATE)
    
    tf.reset_default_graph()
    g1 = tf.Graph()    
    with g1.as_default():
        # Obtain the iterators to each batch of the data
        x_batch, y_batch = inputPipeline()
        # Create the forward computation graph based on the iterators
        y_cap = emiLSTM(x_batch)
        # Create loss graphs and training routines
        emiTrainer(y_cap, y_batch)
        
    #------------------------------END OF COMPUTATION GRAPH------------------------------#
    
    #-------------------------------------EMI DRIVER-------------------------------------#
        
    with g1.as_default():
        emiDriver = EMI_Driver(inputPipeline, emiLSTM, emiTrainer)

    emiDriver.initializeSession(g1)
    y_updated, modelStats = emiDriver.run(numClasses=NUM_OUTPUT, x_train=x_train,
                                          y_train=y_train, bag_train=BAG_TRAIN,
                                          x_val=x_val, y_val=y_val, bag_val=BAG_VAL,
                                          numIter=NUM_ITER, keep_prob=KEEP_PROB,
                                          numRounds=NUM_ROUNDS, batchSize=BATCH_SIZE,
                                          numEpochs=NUM_EPOCHS, modelPrefix=MODEL_PREFIX,
                                          fracEMI=0.5, updatePolicy='top-k', k=1)
    
    #-------------------------------END OF EMI DRIVER-------------------------------------#
    
    #-----------------------------------EARLY SAVINGS-------------------------------------#
    
    """
        Early Prediction Policy: We make an early prediction based on the predicted classes
        probability. If the predicted class probability > minProb at some step, we make
        a prediction at that step.
    """
    def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
        assert instanceOut.ndim == 2
        classes = np.argmax(instanceOut, axis=1)
        prob = np.max(instanceOut, axis=1)
        index = np.where(prob >= minProb)[0]
        if len(index) == 0:
            assert (len(instanceOut) - 1) == (len(classes) - 1)
            return classes[-1], len(instanceOut) - 1
        index = index[0]
        return classes[index], index

    def getEarlySaving(predictionStep, numTimeSteps, returnTotal=False):
        predictionStep = predictionStep + 1
        predictionStep = np.reshape(predictionStep, -1)
        totalSteps = np.sum(predictionStep)
        maxSteps = len(predictionStep) * numTimeSteps
        savings = 1.0 - (totalSteps / maxSteps)
        if returnTotal:
            return savings, totalSteps
        return savings
    
    #--------------------------------END OF EARLY SAVINGS---------------------------------#
    
    #----------------------------------------BEST MODEL-----------------------------------#
    
    k = 2
    predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                   minProb=0.99, keep_prob=1.0)
    bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
    print('Accuracy at k = %d: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
    mi_savings = (1 - NUM_TIMESTEPS / ORIGINAL_NUM_TIMESTEPS)
    emi_savings = getEarlySaving(predictionStep, NUM_TIMESTEPS)
    total_savings = mi_savings + (1 - mi_savings) * emi_savings
    print('Savings due to MI-RNN : %f' % mi_savings)
    print('Savings due to Early prediction: %f' % emi_savings)
    print('Total Savings: %f' % (total_savings))
    
    #Store in the dictionary.
    lstm_dict["k"] = k
    lstm_dict["accuracy"] = np.mean((bagPredictions == BAG_TEST).astype(int))
    lstm_dict["total_savings"] = total_savings
    lstm_dict["y_test"] = BAG_TEST
    lstm_dict["y_pred"] = bagPredictions
    
    # A slightly more detailed analysis method is provided. 
    df = emiDriver.analyseModel(predictions, BAG_TEST, NUM_SUBINSTANCE, NUM_OUTPUT)
#     print (tabulate(df, headers=list(df.columns), tablefmt='grid'))
    
    lstm_dict["detailed analysis"] = df
    #----------------------------------END OF BEST MODEL-----------------------------------#
    
    #----------------------------------PICKING THE BEST MODEL------------------------------#
    
    devnull = open(os.devnull, 'r')
    for val in modelStats:
        round_, acc, modelPrefix, globalStep = val
        emiDriver.loadSavedGraphToNewSession(modelPrefix, globalStep, redirFile=devnull)
        predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                   minProb=0.99, keep_prob=1.0)

        bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
        print("Round: %2d, Validation accuracy: %.4f" % (round_, acc), end='')
        print(', Test Accuracy (k = %d): %f, ' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))), end='')
        print('Additional savings: %f' % getEarlySaving(predictionStep, NUM_TIMESTEPS)) 
        
    
    #-------------------------------END OF PICKING THE BEST MODEL--------------------------#

    return lstm_dict

def experiment_generator(params, path, model = 'lstm'):
    
    if (model == 'lstm'): return lstm_experiment_generator(params, path)
    elif (model == 'fastgrnn'): return fastgrnn_experiment_generator(params, path)
    elif (model == 'gru'): return gru_experiment_generator(params, path)
    elif (model == 'baseline'): return baseline_experiment_generator(params, path)
    
    return 
def gru_experiment_generator(params, data, path = './DSAAR/64_16/'):
    """
        Function that will generate the experiments to be run.
        Inputs : 
        (1) Dictionary params, to set the network parameters.
        (2) Name of the Model to be run from [EMI-LSTM, EMI-FastGRNN, EMI-GRU]
        (3) Path to the dataset, where the csv files are present.
    """
    
    #Copy the params into the gru_dict.
    gru_dict = {**params}
    
    #---------------------------PARAM SETTING----------------------#
    
    # Network parameters for our LSTM + FC Layer
    NUM_HIDDEN = params["NUM_HIDDEN"]
    NUM_TIMESTEPS = params["NUM_TIMESTEPS"]
    ORIGINAL_NUM_TIMESTEPS = params["ORIGINAL_NUM_TIMESTEPS"]
    NUM_FEATS = params["NUM_FEATS"]
    FORGET_BIAS = params["FORGET_BIAS"]
    NUM_OUTPUT = params["NUM_OUTPUT"]
    USE_DROPOUT = True if (params["USE_DROPOUT"] == 1) else False
    KEEP_PROB = params["KEEP_PROB"]

    # For dataset API
    PREFETCH_NUM = params["PREFETCH_NUM"]
    BATCH_SIZE = params["BATCH_SIZE"]

    # Number of epochs in *one iteration*
    NUM_EPOCHS = params["NUM_EPOCHS"]
    # Number of iterations in *one round*. After each iteration,
    # the model is dumped to disk. At the end of the current
    # round, the best model among all the dumped models in the
    # current round is picked up..
    NUM_ITER = params["NUM_ITER"]
    # A round consists of multiple training iterations and a belief
    # update step using the best model from all of these iterations
    NUM_ROUNDS = params["NUM_ROUNDS"]
    LEARNING_RATE = params["LEARNING_RATE"]

    # A staging direcory to store models
    MODEL_PREFIX = params["MODEL_PREFIX"]
    
    #----------------------END OF PARAM SETTING----------------------#
    
    #----------------------DATA LOADING------------------------------#
    
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    x_val, y_val = data['x_val'], data['y_val']
    
    
    # BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
    # step of EMI/MI RNN
    BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)
    BAG_TRAIN = np.argmax(y_train[:, 0, :], axis=1)
    BAG_VAL = np.argmax(y_val[:, 0, :], axis=1)
    NUM_SUBINSTANCE = x_train.shape[1]
    print("x_train shape is:", x_train.shape)
    print("y_train shape is:", y_train.shape)
    print("x_test shape is:", x_val.shape)
    print("y_test shape is:", y_val.shape)
    
    #----------------------END OF DATA LOADING------------------------------#    
    
    #----------------------COMPUTATION GRAPH--------------------------------#
    
    # Define the linear secondary classifier
    def createExtendedGraph(self, baseOutput, *args, **kwargs):
        W1 = tf.Variable(np.random.normal(size=[NUM_HIDDEN, NUM_OUTPUT]).astype('float32'), name='W1')
        B1 = tf.Variable(np.random.normal(size=[NUM_OUTPUT]).astype('float32'), name='B1')
        y_cap = tf.add(tf.tensordot(baseOutput, W1, axes=1), B1, name='y_cap_tata')
        self.output = y_cap
        self.graphCreated = True

    def restoreExtendedGraph(self, graph, *args, **kwargs):
        y_cap = graph.get_tensor_by_name('y_cap_tata:0')
        self.output = y_cap
        self.graphCreated = True

    def feedDictFunc(self, keep_prob=None, inference=False, **kwargs):
        if inference is False:
            feedDict = {self._emiGraph.keep_prob: keep_prob}
        else:
            feedDict = {self._emiGraph.keep_prob: 1.0}
        return feedDict

    EMI_GRU._createExtendedGraph = createExtendedGraph
    EMI_GRU._restoreExtendedGraph = restoreExtendedGraph

    if USE_DROPOUT is True:
        EMI_Driver.feedDictFunc = feedDictFunc
    
    inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
    emiGRU = EMI_GRU(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS,
                            useDropout=USE_DROPOUT)
    emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy',
                             stepSize=LEARNING_RATE)

    tf.reset_default_graph()
    g1 = tf.Graph()    
    with g1.as_default():
        # Obtain the iterators to each batch of the data
        x_batch, y_batch = inputPipeline()
        # Create the forward computation graph based on the iterators
        y_cap = emiGRU(x_batch)
        # Create loss graphs and training routines
        emiTrainer(y_cap, y_batch)
        
    #------------------------------END OF COMPUTATION GRAPH------------------------------#
    
    #-------------------------------------EMI DRIVER-------------------------------------#
        
    with g1.as_default():
        emiDriver = EMI_Driver(inputPipeline, emiGRU, emiTrainer)

    emiDriver.initializeSession(g1)
    y_updated, modelStats = emiDriver.run(numClasses=NUM_OUTPUT, x_train=x_train,
                                          y_train=y_train, bag_train=BAG_TRAIN,
                                          x_val=x_val, y_val=y_val, bag_val=BAG_VAL,
                                          numIter=NUM_ITER, keep_prob=KEEP_PROB,
                                          numRounds=NUM_ROUNDS, batchSize=BATCH_SIZE,
                                          numEpochs=NUM_EPOCHS, modelPrefix=MODEL_PREFIX,
                                          fracEMI=0.5, updatePolicy='top-k', k=1)

    #-------------------------------END OF EMI DRIVER-------------------------------------#
    
    #-----------------------------------EARLY SAVINGS-------------------------------------#
    
    """
        Early Prediction Policy: We make an early prediction based on the predicted classes
        probability. If the predicted class probability > minProb at some step, we make
        a prediction at that step.
    """
    def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
        assert instanceOut.ndim == 2
        classes = np.argmax(instanceOut, axis=1)
        prob = np.max(instanceOut, axis=1)
        index = np.where(prob >= minProb)[0]
        if len(index) == 0:
            assert (len(instanceOut) - 1) == (len(classes) - 1)
            return classes[-1], len(instanceOut) - 1
        index = index[0]
        return classes[index], index

    def getEarlySaving(predictionStep, numTimeSteps, returnTotal=False):
        predictionStep = predictionStep + 1
        predictionStep = np.reshape(predictionStep, -1)
        totalSteps = np.sum(predictionStep)
        maxSteps = len(predictionStep) * numTimeSteps
        savings = 1.0 - (totalSteps / maxSteps)
        if returnTotal:
            return savings, totalSteps
        return savings
    
    #--------------------------------END OF EARLY SAVINGS---------------------------------#
    
    #----------------------------------------BEST MODEL-----------------------------------#
    
    k = 2
    predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                   minProb=0.99, keep_prob=1.0)
    bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
    print('Accuracy at k = %d: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
    mi_savings = (1 - NUM_TIMESTEPS / ORIGINAL_NUM_TIMESTEPS)
    emi_savings = getEarlySaving(predictionStep, NUM_TIMESTEPS)
    total_savings = mi_savings + (1 - mi_savings) * emi_savings
    print('Savings due to MI-RNN : %f' % mi_savings)
    print('Savings due to Early prediction: %f' % emi_savings)
    print('Total Savings: %f' % (total_savings))
    
    # A slightly more detailed analysis method is provided. 
    df = emiDriver.analyseModel(predictions, BAG_TEST, NUM_SUBINSTANCE, NUM_OUTPUT)
#     print (tabulate(df, headers=list(df.columns), tablefmt='grid'))
    
    gru_dict["k"] = k
    gru_dict["accuracy"] = np.mean((bagPredictions == BAG_TEST).astype(int))
    gru_dict["total_savings"] = total_savings
    gru_dict["detailed analysis"] = df
    gru_dict["y_test"] = BAG_TEST
    gru_dict["y_pred"] = bagPredictions
    
    #----------------------------------END OF BEST MODEL-----------------------------------#
    
    #----------------------------------PICKING THE BEST MODEL------------------------------#
    
    devnull = open(os.devnull, 'r')
    for val in modelStats:
        round_, acc, modelPrefix, globalStep = val
        emiDriver.loadSavedGraphToNewSession(modelPrefix, globalStep, redirFile=devnull)
        predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                   minProb=0.99, keep_prob=1.0)

        bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
        print("Round: %2d, Validation accuracy: %.4f" % (round_, acc), end='')
        print(', Test Accuracy (k = %d): %f, ' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))), end='')
        mi_savings = (1 - NUM_TIMESTEPS / ORIGINAL_NUM_TIMESTEPS)
        emi_savings = getEarlySaving(predictionStep, NUM_TIMESTEPS)
        total_savings = mi_savings + (1 - mi_savings) * emi_savings
        print("Total Savings: %f" % total_savings)
        
    #-------------------------------END OF PICKING THE BEST MODEL--------------------------#

    return gru_dict

def fastgrnn_experiment_generator(params,data, path = './HAR/48_16/'):
    """
        Function that will generate the experiments to be run.
        Inputs : 
        (1) Dictionary params, to set the network parameters.
        (2) Name of the Model to be run from [EMI-LSTM, EMI-FastGRNN, EMI-GRU]
        (3) Path to the dataset, where the csv files are present.
    """
    
    #Copy the params to the fastrgnn_dict.
    fastgrnn_dict = {**params}
    
    #---------------------------PARAM SETTING----------------------#
    
    # Network parameters for our FastGRNN + FC Layer
    NUM_HIDDEN = params["NUM_HIDDEN"]
    NUM_TIMESTEPS = params["NUM_TIMESTEPS"]
    NUM_FEATS = params["NUM_FEATS"]
    FORGET_BIAS = params["FORGET_BIAS"]
    NUM_OUTPUT = params["NUM_OUTPUT"]
    USE_DROPOUT = True if (params["USE_DROPOUT"] == 1) else 0
    KEEP_PROB = params["KEEP_PROB"]

    # Non-linearities can be chosen among "tanh, sigmoid, relu, quantTanh, quantSigm"
    UPDATE_NL = params["UPDATE_NL"]
    GATE_NL = params["GATE_NL"]

    # Ranks of Parameter matrices for low-rank parameterisation to compress models.
    WRANK = params["WRANK"]
    URANK = params["URANK"]

    # For dataset API
    PREFETCH_NUM = params["PREFETCH_NUM"]
    BATCH_SIZE = params["BATCH_SIZE"]

    # Number of epochs in *one iteration*
    NUM_EPOCHS = params["NUM_EPOCHS"]
    # Number of iterations in *one round*. After each iteration,
    # the model is dumped to disk. At the end of the current
    # round, the best model among all the dumped models in the
    # current round is picked up..
    NUM_ITER = params["NUM_ITER"]
    # A round consists of multiple training iterations and a belief
    # update step using the best model from all of these iterations
    NUM_ROUNDS = params["NUM_ROUNDS"]

    # A staging direcory to store models
    MODEL_PREFIX = params["MODEL_PREFIX"]
    
    #----------------------END OF PARAM SETTING----------------------#
    
    #----------------------DATA LOADING------------------------------#
    
    # Loading the data
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    x_val, y_val = data['x_val'], data['y_val']
    
    # BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
    # step of EMI/MI RNN
    BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)
    BAG_TRAIN = np.argmax(y_train[:, 0, :], axis=1)
    BAG_VAL = np.argmax(y_val[:, 0, :], axis=1)
    NUM_SUBINSTANCE = x_train.shape[1]
    print("x_train shape is:", x_train.shape)
    print("y_train shape is:", y_train.shape)
    print("x_test shape is:", x_val.shape)
    print("y_test shape is:", y_val.shape)
    
    #----------------------END OF DATA LOADING------------------------------#    
    
    #----------------------COMPUTATION GRAPH--------------------------------#
    
    # Define the linear secondary classifier
    def createExtendedGraph(self, baseOutput, *args, **kwargs):
        W1 = tf.Variable(np.random.normal(size=[NUM_HIDDEN, NUM_OUTPUT]).astype('float32'), name='W1')
        B1 = tf.Variable(np.random.normal(size=[NUM_OUTPUT]).astype('float32'), name='B1')
        y_cap = tf.add(tf.tensordot(baseOutput, W1, axes=1), B1, name='y_cap_tata')
        self.output = y_cap
        self.graphCreated = True

    def restoreExtendedGraph(self, graph, *args, **kwargs):
        y_cap = graph.get_tensor_by_name('y_cap_tata:0')
        self.output = y_cap
        self.graphCreated = True

    def feedDictFunc(self, keep_prob=None, inference=False, **kwargs):
        if inference is False:
            feedDict = {self._emiGraph.keep_prob: keep_prob}
        else:
            feedDict = {self._emiGraph.keep_prob: 1.0}
        return feedDict


    EMI_FastGRNN._createExtendedGraph = createExtendedGraph
    EMI_FastGRNN._restoreExtendedGraph = restoreExtendedGraph
    if USE_DROPOUT is True:
        EMI_FastGRNN.feedDictFunc = feedDictFunc
        
    inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
    emiFastGRNN = EMI_FastGRNN(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS, wRank=WRANK, uRank=URANK, 
                               gate_non_linearity=GATE_NL, update_non_linearity=UPDATE_NL, useDropout=USE_DROPOUT)
    emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy')

    tf.reset_default_graph()
    g1 = tf.Graph()    
    with g1.as_default():
        # Obtain the iterators to each batch of the data
        x_batch, y_batch = inputPipeline()
        # Create the forward computation graph based on the iterators
        y_cap = emiFastGRNN(x_batch)
        # Create loss graphs and training routines
        emiTrainer(y_cap, y_batch)
        
    #------------------------------END OF COMPUTATION GRAPH------------------------------#
    
    #-------------------------------------EMI DRIVER-------------------------------------#
        
    with g1.as_default():
        emiDriver = EMI_Driver(inputPipeline, emiFastGRNN, emiTrainer)

    emiDriver.initializeSession(g1)
    y_updated, modelStats = emiDriver.run(numClasses=NUM_OUTPUT, x_train=x_train,
                                          y_train=y_train, bag_train=BAG_TRAIN,
                                          x_val=x_val, y_val=y_val, bag_val=BAG_VAL,
                                          numIter=NUM_ITER, keep_prob=KEEP_PROB,
                                          numRounds=NUM_ROUNDS, batchSize=BATCH_SIZE,
                                          numEpochs=NUM_EPOCHS, modelPrefix=MODEL_PREFIX,
                                          fracEMI=0.5, updatePolicy='top-k', k=1)

    #-------------------------------END OF EMI DRIVER-------------------------------------#
    
    #-----------------------------------EARLY SAVINGS-------------------------------------#
    
    """
        Early Prediction Policy: We make an early prediction based on the predicted classes
        probability. If the predicted class probability > minProb at some step, we make
        a prediction at that step.
    """
    # Early Prediction Policy: We make an early prediction based on the predicted classes
    #     probability. If the predicted class probability > minProb at some step, we make
    #     a prediction at that step.
    def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
        assert instanceOut.ndim == 2
        classes = np.argmax(instanceOut, axis=1)
        prob = np.max(instanceOut, axis=1)
        index = np.where(prob >= minProb)[0]
        if len(index) == 0:
            assert (len(instanceOut) - 1) == (len(classes) - 1)
            return classes[-1], len(instanceOut) - 1
        index = index[0]
        return classes[index], index

    def getEarlySaving(predictionStep, numTimeSteps, returnTotal=False):
        predictionStep = predictionStep + 1
        predictionStep = np.reshape(predictionStep, -1)
        totalSteps = np.sum(predictionStep)
        maxSteps = len(predictionStep) * numTimeSteps
        savings = 1.0 - (totalSteps / maxSteps)
        if returnTotal:
            return savings, totalSteps
        return savings
    
    #--------------------------------END OF EARLY SAVINGS---------------------------------#
    
    #----------------------------------------BEST MODEL-----------------------------------#
    
    k = 2
    predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb, minProb=0.99)
    bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
    print('Accuracy at k = %d: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
    print('Additional savings: %f' % getEarlySaving(predictionStep, NUM_TIMESTEPS))
    
    # A slightly more detailed analysis method is provided. 
    df = emiDriver.analyseModel(predictions, BAG_TEST, NUM_SUBINSTANCE, NUM_OUTPUT)    
    #print (tabulate(df, headers=list(df.columns), tablefmt='grid'))
    
    fastgrnn_dict["k"] = k
    fastgrnn_dict["accuracy"] = np.mean((bagPredictions == BAG_TEST).astype(int))
    fastgrnn_dict["additional savings"] = getEarlySaving(predictionStep, NUM_TIMESTEPS)
    fastgrnn_dict["detailed analysis"] = df
    fastgrnn_dict["y_test"] = BAG_TEST
    fastgrnn_dict["y_pred"] = bagPredictions
    
    #----------------------------------END OF BEST MODEL-----------------------------------#
    
    
    #----------------------------------PICKING THE BEST MODEL------------------------------#
    
    devnull = open(os.devnull, 'r')
    for val in modelStats:
        round_, acc, modelPrefix, globalStep = val
        emiDriver.loadSavedGraphToNewSession(modelPrefix, globalStep, redirFile=devnull)
        predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                   minProb=0.99, keep_prob=1.0)

        bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
        print("Round: %2d, Validation accuracy: %.4f" % (round_, acc), end='')
        print(', Test Accuracy (k = %d): %f, ' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))), end='')
        print('Additional savings: %f' % getEarlySaving(predictionStep, NUM_TIMESTEPS)) 
        
    
    #-------------------------------END OF PICKING THE BEST MODEL--------------------------#

    return fastgrnn_dict

def emirnn_preprocess(bagsize,no_of_features,extractedDir,numClass,subinstanceLen, subinstanceStride,data_csv,raw_create):
    """
    emirnn_preprocess(bagsize,no_of_features,extractedDir,numClass,subinstanceLen, subinstanceStride,data_csv,raw_create)
    
    extractedDir: location where the raw file and output models have to be stored
    
    data_csv: the extracted hof file.
    
    raw_create: 1 if  raw file already exists in directory 
               0 if  need to create file 
    
    """
    import pandas as pd
    import numpy as np
    import os 
    import video_stuff
    numSteps=bagsize
    numFeats=no_of_features
    try:
        dataset_name = pd.read_csv(data_csv,index_col=0)
    except:
        dataset_name = data_csv
    labels = dataset_name['label']


    labels=pd.DataFrame(labels)

    dataset_name.drop(['video_path','label'],axis=1,inplace=True)
    filtered_train = dataset_name
    filtered_target = labels

    
    print(filtered_train.shape)
    print(filtered_target.shape)


    y = filtered_target.values.reshape(int(len(filtered_target)/bagsize),bagsize)    #input bagsize

    x = filtered_train.values

                                                                                      #no_of_features=540
    x = x.reshape(int(len(x) / bagsize),bagsize, no_of_features)  


    print(x.shape)                         #(Bags, Timesteps, Features)



    one_hot_list = []
    for i in range(len(y)):
        one_hot_list.append(set(y[i]).pop())

    categorical_y_ver = one_hot_list
    categorical_y_ver = np.array(categorical_y_ver)


    def one_hot(y, numOutput):
        y = np.reshape(y, [-1])
        ret = np.zeros([y.shape[0], numOutput])
        for i, label in enumerate(y):
            ret[i, label] = 1
        return ret


    from sklearn.model_selection import train_test_split
    import pathlib


    x_train_val_combined, x_test, y_train_val_combined, y_test = train_test_split(x, categorical_y_ver, test_size=0.20, random_state=13)


                                                                                            #extractedDir = '/home/adithyapa4444_gmail_com/'

    timesteps = x_train_val_combined.shape[-2] #125
    feats = x_train_val_combined.shape[-1]  #9

    trainSize = int(x_train_val_combined.shape[0]*0.9) #6566
    x_train, x_val = x_train_val_combined[:trainSize], x_train_val_combined[trainSize:] 
    y_train, y_val = y_train_val_combined[:trainSize], y_train_val_combined[trainSize:]

    # normalization
    x_train = np.reshape(x_train, [-1, feats])
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    # normalize train
    x_train = x_train - mean
    x_train = x_train / std
    x_train = np.reshape(x_train, [-1, timesteps, feats])

    # normalize val
    x_val = np.reshape(x_val, [-1, feats])
    x_val = x_val - mean
    x_val = x_val / std
    x_val = np.reshape(x_val, [-1, timesteps, feats])

    # normalize test
    x_test = np.reshape(x_test, [-1, feats])
    x_test = x_test - mean
    x_test = x_test / std
    x_test = np.reshape(x_test, [-1, timesteps, feats])

    # shuffle test, as this was remaining
    idx = np.arange(len(x_test))
    np.random.shuffle(idx)
    x_test = x_test[idx]
    y_test = y_test[idx]
    extractedDir += '/'
    if raw_create==0:
        if (os.path.isdir(extractedDir+'RAW')):
            print("A raw file already exsist in this directory")
            return
        else:
            numOutput = numClass
            y_train = one_hot(y_train, numOutput)
            y_val = one_hot(y_val, numOutput)
            y_test = one_hot(y_test, numOutput)
            

            pathlib.Path(extractedDir + 'RAW').mkdir(parents=True, exist_ok = True)

            np.save(extractedDir + "RAW/x_train", x_train)
            np.save(extractedDir + "RAW/y_train", y_train)
            np.save(extractedDir + "RAW/x_test", x_test)
            np.save(extractedDir + "RAW/y_test", y_test)
            np.save(extractedDir + "RAW/x_val", x_val)
            np.save(extractedDir + "RAW/y_val", y_val)
    numOutput = numClass
    y_train = one_hot(y_train, numOutput)
    y_val = one_hot(y_val, numOutput)
    y_test = one_hot(y_test, numOutput)

    data = {
        'x_train' : x_train, 
        'y_train' : y_train, 
        'x_val' : x_val, 
        'y_val' : y_val, 
        'x_test' : x_test, 
        'y_test' : y_test
    }
            
        

    def loadData(dirname):
        x_train = np.load(dirname + '/' + 'x_train.npy')
        y_train = np.load(dirname + '/' + 'y_train.npy')
        x_test = np.load(dirname + '/' + 'x_test.npy')
        y_test = np.load(dirname + '/' + 'y_test.npy')
        x_val = np.load(dirname + '/' + 'x_val.npy')
        y_val = np.load(dirname + '/' + 'y_val.npy')
        return x_train, y_train, x_test, y_test, x_val, y_val


    def bagData(X, Y, subinstanceLen, subinstanceStride,numClass,numSteps,numFeats):
        '''
        Takes x and y of shape
        [-1, 128, 9] and [-1, 6] respectively and converts it into bags of instances.
        returns [-1, numInstance, ]
        '''
        #numClass = 2
        #numSteps = 24 # Window length
        #numFeats = 540 # Number of features
        print("subinstanceLen:",subinstanceLen,"\nsubinstanceStride :", subinstanceStride,"\nnumClass:",numClass,"\nnumSteps",numSteps,"\nnumFeats",numFeats)
        print("X Shape :",X.shape)
        print("Y Shape :",Y.shape)
        assert X.ndim == 3
        assert X.shape[1] == numSteps
        assert X.shape[2] == numFeats
        assert subinstanceLen <= numSteps
        assert subinstanceLen > 0 # subinstance length = Number of readings for which the class signature occurs
        assert subinstanceStride <= numSteps  
        assert subinstanceStride >= 0 
        assert len(X) == len(Y)
        assert Y.ndim == 2
        assert Y.shape[1] == numClass
        x_bagged = []
        y_bagged = []
        for i, point in enumerate(X[:, :, :]):
            instanceList = []
            start = 0
            end = subinstanceLen
            while True:
                x = point[start:end, :]
                if len(x) < subinstanceLen:
                    x_ = np.zeros([subinstanceLen, x.shape[1]])
                    x_[:len(x), :] = x[:, :]
                    x = x_
                instanceList.append(x)
                if end >= numSteps:
                    break
                start += subinstanceStride
                end += subinstanceStride
            bag = np.array(instanceList)
            numSubinstance = bag.shape[0]
            label = Y[i]
            label = np.argmax(label)
            labelBag = np.zeros([numSubinstance, numClass])
            labelBag[:, label] = 1
            x_bagged.append(bag)
            label = np.array(labelBag)
            y_bagged.append(label)
        return np.array(x_bagged), np.array(y_bagged)

                                                                                                            #sourceDir, outDir

    def makeEMIData(subinstanceLen, subinstanceStride, data, sourceDir, outDir,numClass,numSteps,numFeats):
        x_train, y_train, x_test, y_test, x_val, y_val = loadData(sourceDir)
        try:
            x_train, y_train = bagData(x_train, y_train, subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
    #         np.save(outDir + '/x_train.npy', x)
    #         np.save(outDir + '/y_train.npy', y)
            print('Num train %d' % len(x))
            x_test, y_test = bagData(x_test, y_test, subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
    #         np.save(outDir + '/x_test.npy', x)
    #         np.save(outDir + '/y_test.npy', y)
            print('Num test %d' % len(x))
            x_val, y_val = bagData(x_val, y_val, subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
            print('Num val %d' % len(x))
        except:
            x_train, y_train = bagData(data['x_train'], data['y_train'], subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
    #         np.save(outDir + '/x_train.npy', x)
    #         np.save(outDir + '/y_train.npy', y)
            print('Num train %d' % len(x))
            x_test, y_test = bagData(data['x_test'], data['y_test'], subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
    #         np.save(outDir + '/x_test.npy', x)
    #         np.save(outDir + '/y_test.npy', y)
            print('Num test %d' % len(x))
            x_val, y_val = bagData(data['x_val'], data['y_val'], subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
            print('Num val %d' % len(x))
#         np.save(outDir + '/x_val.npy', x)
#         np.save(outDir + '/y_val.npy', y)
        
        return(x_train, y_train, x_val, y_val, x_test, y_test)



                                                                                        #subinstanceLen = 12
                                                                                        #subinstanceStride = 3
                                                                                        #extractedDir = '/home/adithyapa4444_gmail_com'
    rawDir = extractedDir + 'RAW'
    sourceDir = rawDir
    from os import mkdir
    # WHEN YOU CHANGE THE ABOVE - CREATE A FOLDER 
    if (not(os.path.isdir(extractedDir+'/'+str(subinstanceLen)+'_'+str(subinstanceStride)))):
        mkdir(extractedDir+'/'+str(subinstanceLen)+'_'+str(subinstanceStride))  

    outDir = extractedDir + '%d_%d' % (subinstanceLen, subinstanceStride)
    return(makeEMIData(subinstanceLen, subinstanceStride, data, sourceDir, outDir,numClass,numSteps,numFeats))

from sklearn.metrics import classification_report, confusion_matrix
import time
def metrics_fastgrnn(model_prefix, x_test, y_test, model_no):
    def createExtendedGraph(self, baseOutput, *args, **kwargs):
        W1 = tf.Variable(np.random.normal(size=[NUM_HIDDEN, NUM_OUTPUT]).astype('float32'), name='W1')
        B1 = tf.Variable(np.random.normal(size=[NUM_OUTPUT]).astype('float32'), name='B1')
        y_cap = tf.add(tf.tensordot(baseOutput, W1, axes=1), B1, name='y_cap_tata')
        self.output = y_cap
        self.graphCreated = True

    def addExtendedAssignOps(self, graph, W_val=None, B_val=None):
        W1 = graph.get_tensor_by_name('W1:0')
        B1 = graph.get_tensor_by_name('B1:0')
        W1_op = tf.assign(W1, W_val)
        B1_op = tf.assign(B1, B_val)
        self.assignOps.extend([W1_op, B1_op])

    def restoreExtendedGraph(self, graph, *args, **kwargs):
        y_cap = graph.get_tensor_by_name('y_cap_tata:0')
        self.output = y_cap
        self.graphCreated = True

    def feedDictFunc(self, keep_prob, **kwargs):
        feedDict = {self._emiGraph.keep_prob: keep_prob}
        return feedDict

    EMI_FastGRNN._createExtendedGraph = createExtendedGraph
    EMI_FastGRNN._restoreExtendedGraph = restoreExtendedGraph
    EMI_FastGRNN.addExtendedAssignOps = addExtendedAssignOps

    def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
        assert instanceOut.ndim == 2
        classes = np.argmax(instanceOut, axis=1)
        prob = np.max(instanceOut, axis=1)
        index = np.where(prob >= minProb)[0]
        if len(index) == 0:
            assert (len(instanceOut) - 1) == (len(classes) - 1)
            return classes[-1], len(instanceOut) - 1
        index = index[0]
        return classes[index], index
    NUM_HIDDEN = 128
    NUM_TIMESTEPS = x_test.shape[-2]
    NUM_SUBINSTANCE=x_test.shape[1]
    NUM_FEATS = x_test.shape[-1]
    FORGET_BIAS = 1.0
    NUM_OUTPUT = y_test.shape[-1]
    USE_DROPOUT = 0

    KEEP_PROB = 0.9
    UPDATE_NL = "quantTanh"
    GATE_NL = "quantSigm"
    WRANK = 5
    URANK = 6
    PREFETCH_NUM = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    NUM_ITER = 3
    NUM_ROUNDS = 4 

    if USE_DROPOUT is True:
        EMI_Driver.feedDictFunc = feedDictFunc
    
    def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
        assert instanceOut.ndim == 2
        classes = np.argmax(instanceOut, axis=1)
        prob = np.max(instanceOut, axis=1)
        index = np.where(prob >= minProb)[0]
        if len(index) == 0:
            assert (len(instanceOut) - 1) == (len(classes) - 1)
            return classes[-1], len(instanceOut) - 1
        index = index[0]
        return classes[index], index
    

    
    tf.reset_default_graph()
    sess = tf.Session()
    graphManager = utils.GraphManager()
    graph = graphManager.loadCheckpoint(sess, model_prefix, globalStep=model_no)
    inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT, graph=graph)
    emiLSTM = EMI_FastGRNN(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS, wRank=WRANK, uRank=URANK, 
                                   gate_non_linearity=GATE_NL, update_non_linearity=UPDATE_NL, useDropout=USE_DROPOUT)
    emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy', graph=graph)

    g1 = graph
    with g1.as_default():
        x_batch, y_batch = inputPipeline()
        y_cap = emiLSTM(x_batch)
        emiTrainer(y_cap, y_batch)

    with g1.as_default():
        emiDriver = EMI_Driver(inputPipeline, emiLSTM, emiTrainer)
    emiDriver.setSession(sess)
    k = 2
    for _ in range(3):
        start = time.time()
        predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                    minProb=0.99, keep_prob=1.0)
        bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)

        end = time.time()
    y_test_bag = np.argmax(y_test[:, 0, :], axis=1)
    metrics_dict = {'time_run': end - start,
                    'y_pred': bagPredictions,
                    'y_test': y_test_bag,
                    'classification_report': classification_report(bagPredictions, y_test_bag),
                    'confusion_matrix': confusion_matrix(bagPredictions, y_test_bag)}
    return metrics_dict

reduction_factors = [4,3,2]
bagsizes = [32,43,64]
subinst = [16,22,32]
stride = [4,6,8]
from time import time
No_of_features =540
# ExtractedDir = '/home/adithyapa4444_gmail_com/UCF_Crime/half/'
NumClass = 8
ExtractedDir = '/home/adithyapa4444_gmail_com/UCF_Crime/half/'
for rf, bag, si, st in zip(reduction_factors,bagsizes, subinst, stride):
    print(rf, bag, si, st)
    
    #--------------------------------------------------------------------
    file_1 = open('/home/adithyapa4444_gmail_com/FPS_Expt_Results/get_hof_times.txt','a')
    #--------------------------------------------------------------------
    start = time()
    hof_df = get_processed_hof(src_folder='/home/adithyapa4444_gmail_com/UCF_Crime/Full_UCrime/annotated/',bagsize=bag,reduction_factor=rf)
    time_string = "\nBagsize : "+str(bag)+"\nReduction Factor : "+str(rf)+"\nTime :" + str(time()-start) + "\nNumber of bags:" + str(len(hof_df) // bag)
    file_1.write(time_string)
    file_1.close()
    x_train, y_train, x_val, y_val, x_test, y_test = emirnn_preprocess(bagsize=bag,
                      no_of_features=No_of_features,
                      extractedDir=ExtractedDir,
                      numClass=8,
                      subinstanceLen=si,
                      subinstanceStride=st,
                      data_csv=hof_df,
                      raw_create=1
                     )
    del hof_df
    #--------------------------------------------------------------
    data = {
        'x_train' : x_train, 
        'y_train' : y_train, 
        'x_val' : x_val, 
        'y_val' : y_val, 
        'x_test' : x_test, 
        'y_test' : y_test
    }
    # -------------------------------------------------------------
    file_2 = open('/home/adithyapa4444_gmail_com/FPS_Expt_Results/reduced_FPS_metrics.txt','a')
    # ----------------------------------------------------
#     print('############FGRNN for reduction factor',rf)
#     dataset = 'CRIME_FULL_FASTGRNN'
#     model = 'fastgrnn'
#     model_prefix = '/home/adithyapa4444_gmail_com/FPS_Expt_Results/' + model + '_' + str(rf) + '/'
#     os.mkdir(model_prefix)
#     path = ExtractedDir+str(si)+'_'+str(st)+'/'
#     ## fastgrnn
#     fastgrnn_param = {
#         "NUM_HIDDEN" : 128,
#         "NUM_TIMESTEPS" : si,
#         "NUM_FEATS" : 540,
#         "FORGET_BIAS" : 1.0,
#         "NUM_OUTPUT" : 8,
#         "USE_DROPOUT" : 0,

#         "KEEP_PROB" : 0.9,
#         "UPDATE_NL" : "quantTanh",
#         "GATE_NL" : "quantSigm",
#         "WRANK" : 5,
#         "URANK" : 6,
#         "PREFETCH_NUM" : 5,
#         "BATCH_SIZE" : 32,
#         "NUM_EPOCHS" : 2,
#         "NUM_ITER" : 3,
#         "NUM_ROUNDS" : 4,
#         "MODEL_PREFIX" : model_prefix + 'model-' + str(model)
#     }
#     fastgrnn_dict = fastgrnn_experiment_generator(fastgrnn_param, data, path)
#     metrics = metrics_fastgrnn(model_prefix + '/model-' + str(model), data['x_test'], data['y_test'], 1010)
#     file_2.write('\nFASTGRNN_skipped_frame_' + str(rf) + '_' + str(bag) + '\n')
#     for metric in metrics:
#         lstm_push = metric + ' : ' + str(metrics[metric]) + '\n'
#         file_2.write(lstm_push)
    #-----------------------------------------------------------------------------------
    ## lstm
    print('############LSTM for reduction factor',rf)
    dataset = 'CRIME_FULL'
    path = ExtractedDir+str(si)+'_'+str(st)+'/'
    #Choose model from among [lstm, fastgrnn, gru]
    model = 'lstm'
    model_prefix = '/home/adithyapa4444_gmail_com/FPS_Expt_Results/' + model + '_' + str(rf) + '/'
    os.mkdir(model_prefix)
    # Dictionary to set the parameters.
    params = {
        "NUM_HIDDEN" : 128,
        "NUM_TIMESTEPS" : int(si), #subinstance length.
        "ORIGINAL_NUM_TIMESTEPS" : int(bag), # Window length 
        "NUM_FEATS" : 540,
        "FORGET_BIAS" : 1.0,
        "NUM_OUTPUT" : 8,   # Number of target classes
        "USE_DROPOUT" : 1, # '1' -> True. '0' -> False
        "KEEP_PROB" : 0.75,
        "PREFETCH_NUM" : 5,
        "BATCH_SIZE" : 32,
        "NUM_EPOCHS" : 2,
        "NUM_ITER" : 3,
        "NUM_ROUNDS" : 4,
        "LEARNING_RATE" : 0.001,
        "MODEL_PREFIX" : model_prefix + 'model-' + str(model)
    }

    #Preprocess data, and load the train,test and validation splits.
    lstm_dict = lstm_experiment_generator(params, data, path)
    metrics = emi_model_metrics(model_prefix + '/model-' + str(model),data['x_test'], data['y_test'], model_prefix, 1010)
    file_2.write('\nLSTM_skipped_frame_' + str(rf) + '_' + str(bag) + '\n')
    for metric in metrics:
        lstm_push = metric + ' : ' + str(metrics[metric]) + '\n'
        file_2.write(lstm_push)
        
    #--------------------------------------------------------------
    print('############gru for reduction factor',rf)
    ## gru
    dataset = 'CRIME_FULL_GRU'
    path = ExtractedDir+str(si)+'_'+str(st)+'/'

    #Choose model from among [lstm, fastgrnn, gru]
    model = 'gru'
    model_prefix = '/home/adithyapa4444_gmail_com/FPS_Expt_Results/' + model + '_' + str(rf) + '/'
    os.mkdir(model_prefix)
    # Dictionary to set the parameters.
    gru_params = {
        "NUM_HIDDEN" : 128,
        "NUM_TIMESTEPS" : si,
        "ORIGINAL_NUM_TIMESTEPS" : bag,
        "NUM_FEATS" : 540,
        "FORGET_BIAS" : 1.0,
        "NUM_OUTPUT" : 8,
        "USE_DROPOUT" : 1, # '1' -> True. '0' -> False
        "KEEP_PROB" : 0.75,
        "PREFETCH_NUM" : 5,
        "BATCH_SIZE" : 32,
        "NUM_EPOCHS" : 2,
        "NUM_ITER" : 3,
        "NUM_ROUNDS" : 4,
        "LEARNING_RATE" : 0.001,
        "MODEL_PREFIX" : model_prefix + 'model-' + str(model)
    }
    #Preprocess data, and load the train,test and validation splits.
    gru_dict = gru_experiment_generator(gru_params, data, path)
    metrics = emi_model_metrics(model_prefix + '/model-' + str(model),data['x_test'], data['y_test'], model_prefix, 1010)
    file_2.write('\nGRU_skipped_frame_' + str(rf) + '_' + str(bag) + '\n')
    for metric in metrics:
        lstm_push = metric + ' : ' + str(metrics[metric]) + '\n'
        file_2.write(lstm_push)
    
    file_2.write('\n------------------------------------------------------------------------------------------------------\n')
    file_2.close()

