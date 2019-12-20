from __future__ import print_function
import time
import os
import sys
import tensorflow as tf
import numpy as np
from rnn import EMI_DataPipeline
from rnn import EMI_BasicLSTM, EMI_GRU, EMI_FastGRNN
from emirnnTrainer import EMI_Trainer, EMI_Driver
from sklearn.metrics import classification_report, confusion_matrix
import utils
def emi_model_metrics(model_prefix, x_test,y_test, extracted_dir,model_no):
    # Network parameters for our LSTM + FC Layer
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
    k = 2
    NUM_HIDDEN = 128
    NUM_TIMESTEPS = x_test.shape[-2]
    NUM_SUBINSTANCE=x_test.shape[1]
    NUM_FEATS = x_test.shape[-1]
    FORGET_BIAS = 1.0
    NUM_OUTPUT = y_test.shape[-1]
    USE_DROPOUT = True
    KEEP_PROB = 0.75
    PREFETCH_NUM = 5
    BATCH_SIZE = 32
    
#     sess = tf.Session()
#     graphManager = utils.GraphManager()
#     graph = graphManager.loadCheckpoint(sess, model_prefix, globalStep=model_no)

    if model_prefix.split('-')[-1] == 'lstm':
        EMI_BasicLSTM._createExtendedGraph = createExtendedGraph
        EMI_BasicLSTM._restoreExtendedGraph = restoreExtendedGraph
        EMI_BasicLSTM.addExtendedAssignOps = addExtendedAssignOps

    
        if USE_DROPOUT is True:
            EMI_Driver.feedDictFunc = feedDictFunc
            
        tf.reset_default_graph()

        inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
        emiModel = EMI_BasicLSTM(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS,
                                forgetBias=FORGET_BIAS, useDropout=USE_DROPOUT)
        
    elif model_prefix.split('-')[-1] == 'gru':
        EMI_GRU._createExtendedGraph = createExtendedGraph
        EMI_GRU._restoreExtendedGraph = restoreExtendedGraph
        EMI_GRU.addExtendedAssignOps = addExtendedAssignOps

        if USE_DROPOUT is True:
            EMI_Driver.feedDictFunc = feedDictFunc
        tf.reset_default_graph()

        inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
        emiModel = EMI_GRU(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS,
                                useDropout=USE_DROPOUT)
    elif model_prefix.split('-')[-1] == 'fastgrnn':
        USE_DROPOUT = False
        EMI_FastGRNN._createExtendedGraph = createExtendedGraph
        EMI_FastGRNN._restoreExtendedGraph = restoreExtendedGraph
        EMI_FastGRNN.addExtendedAssignOps = addExtendedAssignOps
        
        if USE_DROPOUT is True:
            EMI_Driver.feedDictFunc = feedDictFunc
        tf.reset_default_graph()

        inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
        emiModel = EMI_FastGRNN(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS, wRank=5, uRank=6, 
                                       gate_non_linearity="quantSigm", update_non_linearity="quantTanh", useDropout=USE_DROPOUT)

        
    emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy')


    # For dataset API
    

    # Construct the graph
    g1 = tf.Graph()
    with g1.as_default():
        x_batch, y_batch = inputPipeline()
        y_cap = emiModel(x_batch)
        emiTrainer(y_cap, y_batch)

    with g1.as_default():
        emiDriver = EMI_Driver(inputPipeline, emiModel, emiTrainer)

    emiDriver.initializeSession(g1)
    tf.reset_default_graph()
    emiDriver.loadSavedGraphToNewSession(model_prefix, model_no)
#     emiDriver.setSession(sess)
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
    
    ### Hook up the model size later
    import pickle as pkl
    import datetime
    now = datetime.datetime.now()
    filename = list((str(now.year),"-",str(now.month),"-",str(now.day),"|",str(now.hour),"-",str(now.minute)))
    filename = ''.join(filename)
    pkl.dump(metrics_dict, open(extracted_dir + "/"+model_prefix.split('-')[-1]+"_dict_" + filename + ".pkl",mode='wb'))
    
    return metrics_dict

def model_size(model, num_hidden, num_classes, num_feats):
    """
        model_size(model, num_hidden, num_classes, num_feats)
        
        model - Model that is currently in use[lstm, gru, rnn, fastrnn, fastgrnn]
        num_hidden - Number of hidden layers
        num_classes - Number of classes
        num_feats - Number of input features
    """
    if model == 'lstm':
        return (4*num_hidden*(num_feats+num_hidden+1) + num_hidden*num_classes+num_classes)*4/1024
    elif model == 'gru':
        return (3*num_hidden*(num_feats+num_hidden+1) + num_hidden*num_classes+num_classes)*4/1024
    elif model == 'rnn':
        return (num_hidden*(num_feats+num_hidden+1) + num_hidden*num_classes+num_classes)*4/1024
    elif model == 'fastrnn':
        return (num_hidden*(num_feats+num_hidden+1) + num_hidden*num_classes+num_classes +2)*4/1024
    elif model == 'fastgrnn':
        return (num_hidden*(num_feats+num_hidden+2) + num_hidden*num_classes+num_classes +2)*4/1024