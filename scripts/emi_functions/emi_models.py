import datetime
import pickle as pkl
import pathlib
from rnn import EMI_DataPipeline
from rnn import EMI_BasicLSTM, EMI_FastGRNN, EMI_FastRNN, EMI_GRU
from emirnnTrainer import EMI_Trainer, EMI_Driver
import utils
import os 
import numpy as np
import tensorflow as tf


def lstm_experiment_generator(params, path = './DSAAR/64_16/'):
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
    
    x_train, y_train = np.load(path + 'x_train.npy'), np.load(path + 'y_train.npy')
    x_test, y_test = np.load(path + 'x_test.npy'), np.load(path + 'y_test.npy')
    x_val, y_val = np.load(path + 'x_val.npy'), np.load(path + 'y_val.npy')

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




def gru_experiment_generator(params, path = './DSAAR/64_16/'):
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
    
    x_train, y_train = np.load(path + 'x_train.npy'), np.load(path + 'y_train.npy')
    x_test, y_test = np.load(path + 'x_test.npy'), np.load(path + 'y_test.npy')
    x_val, y_val = np.load(path + 'x_val.npy'), np.load(path + 'y_val.npy')
    
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
    #print (tabulate(df, headers=list(df.columns), tablefmt='grid'))
    
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



def fastgrnn_experiment_generator(params, path = './HAR/48_16/'):
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
    x_train, y_train = np.load(path + 'x_train.npy'), np.load(path + 'y_train.npy')
    x_test, y_test = np.load(path + 'x_test.npy'), np.load(path + 'y_test.npy')
    x_val, y_val = np.load(path + 'x_val.npy'), np.load(path + 'y_val.npy')
    
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

def lstm(dataset,path,num_hidden,subinstance_length,window_length,num_feats,num_class,epoch,rounds):
        # Baseline EMI-LSTM.

    

    #Choose model from among [lstm, fastgrnn, gru]
    model = 'lstm'

    # Dictionary to set the parameters.
    params = {
        "NUM_HIDDEN" : num_hidden,
        "NUM_TIMESTEPS" : subinstance_length,
        "ORIGINAL_NUM_TIMESTEPS" :window_length, 
        "NUM_FEATS" : num_feats,
        "FORGET_BIAS" : 1.0,
        "NUM_OUTPUT" : num_class,   # Number of target classes
        "USE_DROPOUT" : 1, # '1' -> True. '0' -> False
        "KEEP_PROB" : 0.75,
        "PREFETCH_NUM" : 5,
        "BATCH_SIZE" : 32,
        "NUM_EPOCHS" : epoch,
        "NUM_ITER" : 4,
        "NUM_ROUNDS" : rounds,
        "LEARNING_RATE" : 0.001,
        "MODEL_PREFIX" : dataset + '/model-' + str(model)
    }
    import os,datetime

    #Preprocess data, and load the train,test and validation splits.
    lstm_dict = lstm_experiment_generator(params, path)

    #Create the directory to store the results of this run.

    dirname = ""
    dirname = "./Results" + ''.join(dirname) + "/"+dataset+"/"+model
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    print ("Results for this run have been saved at" , dirname, ".")

    now = datetime.datetime.now()
    filename = list((str(now.year),"-",str(now.month),"-",str(now.day),"|",str(now.hour),"-",str(now.minute)))
    filename = ''.join(filename)

    #Save the dictionary containing the params and the results.
    import pickle as pkl
    pkl.dump(lstm_dict,open(dirname + "/lstm_dict_" + filename + ".pkl",mode='wb'))
    
    
    
def gru (dataset,path,num_hidden,subinstance_length,window_length,num_feats,num_class,epoch,rounds):
    

    #Choose model from among [lstm, fastgrnn, gru]
    model = 'gru'

    # Dictionary to set the parameters.
    gru_params = {
        "NUM_HIDDEN" : num_hidden,
        "NUM_TIMESTEPS" : subinstance_length,
        "ORIGINAL_NUM_TIMESTEPS" :window_length, 
        "NUM_FEATS" : num_feats,
        "FORGET_BIAS" : 1.0,
        "NUM_OUTPUT" : num_class,   # Number of target classes
        "USE_DROPOUT" : 1, # '1' -> True. '0' -> False
        "KEEP_PROB" : 0.75,
        "PREFETCH_NUM" : 5,
        "BATCH_SIZE" : 32,
        "NUM_EPOCHS" : epoch,
        "NUM_ITER" : 4,
        "NUM_ROUNDS" : rounds,
        "LEARNING_RATE" : 0.001,
        "MODEL_PREFIX" : dataset + '/model-' + str(model)
    }

    #Preprocess data, and load the train,test and validation splits.
    gru_dict = gru_experiment_generator(gru_params, path)

    #Create the directory to store the results of this run.

    dirname = ""
    dirname = "./Results" + ''.join(dirname) + "/"+dataset+"/"+model
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    print ("Results for this run have been saved at" , dirname, ".")

    now = datetime.datetime.now()
    filename = list((str(now.year),"-",str(now.month),"-",str(now.day),"|",str(now.hour),"-",str(now.minute)))
    filename = ''.join(filename)

    #Save the dictionary containing the params and the results.
    pkl.dump(gru_dict,open(dirname + "/gru_dict_" + filename + ".pkl",mode='wb'))

    
def fastgrnn(dataset,path,num_hidden,subinstance_length,num_feats,num_class,epoch,rounds,wRANK,uRANK):
    #Choose model from among [lstm, fastgrnn, gru]
    model = 'fastgrnn'

    # Dictionary to set the parameters.
    fastgrnn_params = {
        "NUM_HIDDEN" : num_hidden,
        "NUM_TIMESTEPS" : subinstance_length,
        "NUM_FEATS" : num_feats,
        "FORGET_BIAS" : 1.0,
        "NUM_OUTPUT" : num_class,
        "USE_DROPOUT" : 0, # '1' -> True. '0' -> False
        "KEEP_PROB" : 0.9,
        "UPDATE_NL" : "quantTanh",
        "GATE_NL" : "quantSigm",
        "WRANK" : wRANK,
        "URANK" : uRANK,
        "PREFETCH_NUM" : 5,
        "BATCH_SIZE" : 32,
        "NUM_EPOCHS" : epoch,
        "NUM_ITER" : 4,
        "NUM_ROUNDS" : rounds,
        "MODEL_PREFIX" : dataset + '/model-' + str(model)
    }

    #Preprocess data, and load the train,test and validation splits.
    fastgrnn_dict = fastgrnn_experiment_generator(fastgrnn_params, path)

    #Create the directory to store the results of this run.

    dirname = ""
    dirname = "./Results" + ''.join(dirname) + "/"+dataset+"/"+model
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    print ("Results for this run have been saved at" , dirname, ".")

    now = datetime.datetime.now()
    filename = list((str(now.year),"-",str(now.month),"-",str(now.day),"|",str(now.hour),"-",str(now.minute)))
    filename = ''.join(filename)

    #Save the dictionary containing the params and the results.

    pkl.dump(fastgrnn_dict,open(dirname + "/fastgrnn_dict_" + filename + ".pkl",mode='wb'))