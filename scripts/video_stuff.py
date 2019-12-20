
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch.nn as nn
from sklearn.decomposition import PCA 
import cv2
# from torchsummary import summary
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
import torch
import torchvision

import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
#from skimage.feature import hog
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

''' Taken with modifications from Scikit-Image version of HOG '''

''' Fix HOF: last orientation should be 'no motion' cell '''

model_ft = torchvision.models.resnet18(pretrained = True)
my_model = nn.Sequential(*(list(model_ft.children())[:-1]))

def getFlow(imPrev, imNew):
    flow = cv2.calcOpticalFlowFarneback(imPrev, imNew, flow=None, pyr_scale=.5, levels=3, winsize=9, iterations=1, poly_n=3, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

def get_depthFlow(imPrev, imNew):
    # Should actually go much more than 1 pixel!!!
    flow = np.zeros_like(imPrev)+999
    # flow = np.repeat(flow, 2, 2)
    
    # flow[im1==im2,:]=0
    flow[im1==im2]=4
    for x in xrange(1,im1.shape[0]):
        for y in xrange(1,im1.shape[1]):
            if flow[x,y]==999:
                flow[x,y] = np.argmin(im1[x-1:x+2, y-1:y+2]-im2[x-1:x+2, y-1:y+2])
    flow[flow==999] = -2

    flowNew = np.repeat(flow[:,:,np.newaxis], 2, 2)
    flowNew[flow==0,:] = [-1,-1]
    flowNew[flow==1,:] = [-1, 0]
    flowNew[flow==2,:] = [-1, 1]
    flowNew[flow==3,:] = [ 0,-1]
    flowNew[flow==4,:] = [ 0,0]
    flowNew[flow==5,:] = [ 0, 1]
    flowNew[flow==6,:] = [ 1,-1]
    flowNew[flow==7,:] = [ 1, 0]
    flowNew[flow==8,:] = [ 1, 1]
    return flow

def hog2image(hogArray, imageSize=[32,32],orientations=9,pixels_per_cell=(8, 8),cells_per_block=(3, 3)):
    from scipy import sqrt, pi, arctan2, cos, sin
    from skimage import draw

    sy, sx = imageSize
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1    

    hogArray = hogArray.reshape([n_blocksy, n_blocksx, by, bx, orientations])

    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = hogArray[y, x, :]
                orientation_histogram[y:y + by, x:x + bx, :] = block

    radius = min(cx, cy) // 2 - 1
    hog_image = np.zeros((sy, sx), dtype=float)
    for x in range(n_cellsx):
        for y in range(n_cellsy):
            for o in range(orientations):
                centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                dx = int(radius * cos(float(o) / orientations * np.pi))
                dy = int(radius * sin(float(o) / orientations * np.pi))
                # rr, cc = draw.bresenham(centre[0] - dy, centre[1] - dx,
                #                         centre[0] + dy, centre[1] + dx)
                rr, cc = draw.bresenham(centre[0] - dx, centre[1] - dy,\
                                        centre[0] + dx, centre[1] + dy)  
                hog_image[rr, cc] += orientation_histogram[y, x, o]

    return hog_image


def showSplit(splitIm, blocks=[4,3]):
    for x in range(blocks[0]):
        for y in range(blocks[1]):
            i=y*4+x;
            subplot(4,3,i+1)
            imshow(splitIm[:,:,i])


# def splitIm(im, blocks=[4,3]):
#     subSizeX, subSizeY = im.shape / np.array(blocks)
#     newIms = np.empty([im.shape[0]/blocks[0], im.shape[1]/blocks[1], blocks[0]*blocks[1]])
#     for x in xrange(blocks[0]):
#         for y in xrange(blocks[1]):
#             newIms[:,:, x*blocks[1]+y] = im[x*subSizeX:(x+1)*subSizeX,y*subSizeY:(y+1)*subSizeY]

#     return newIms

def splitIm(im, blocks=[4,3]):
    subSizeX, subSizeY = im.shape / np.array(blocks)
    newIms = []
    for x in xrange(blocks[0]):
        for y in xrange(blocks[1]):
            newIms.append(im[x*subSizeX:(x+1)*subSizeX,y*subSizeY:(y+1)*subSizeY, :])

    newIms = np.dstack(newIms)
    return newIms    

def splitHog(im, blocks=[4,3], visualise=False):
    ims = splitIm(im, blocks)

    hogs = []
    hogIms = []
    for i in range(ims.shape[2]):
        if visualise:
            hogArray, hogIm = hog(colorIm_g, visualise=True)
            hogs.append(hogArray)
            hogIms.append(hogArray)
        else:
            hogArray = hog(colorIm_g, visualise=False)
            hogs.append(hogArray)
    
    if visualise:
        return hogs, hogIms
    else:
        return hogs


def hof(flow, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, normalise=False, motion_threshold=1.):

    """Extract Histogram of Optical Flow (HOF) for a given image.
    Key difference between this and HOG is that flow is MxNx2 instead of MxN
    Compute a Histogram of Optical Flow (HOF) by
        1. (optional) global image normalisation
        2. computing the dense optical flow
        3. computing flow histograms
        4. normalising across blocks
        5. flattening into a feature vector
    Parameters
    ----------
    Flow : (M, N) ndarray
        Input image (x and y flow images).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the hof.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.
    static_threshold : threshold for no motion
    Returns
    -------
    newarr : ndarray
        hof for the image as a 1D (flattened) array.
    hof_image : ndarray (if visualise=True)
        A visualisation of the hof image.
    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA
    """
    flow = np.atleast_2d(flow)

    """ 
    -1-
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if flow.ndim < 3:
        raise ValueError("Requires dense flow in both directions")

    if normalise:
        flow = sqrt(flow)

    """ 
    -2-
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if flow.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers in np.diff()
        flow = flow.astype('float')

    gx = np.zeros(flow.shape[:2])
    gy = np.zeros(flow.shape[:2])
    # gx[:, :-1] = np.diff(flow[:,:,1], n=1, axis=1)
    # gy[:-1, :] = np.diff(flow[:,:,0], n=1, axis=0)

    gx = flow[:,:,1]
    gy = flow[:,:,0]



    """ 
    -3-
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / pi) % 180

    sy, sx = flow.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[int(cy / 2):int(cy * n_cellsy):cy, int(cx / 2):int(cx * n_cellsx):cx]
    for i in range(orientations-1):
        #create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        #print(temp_filt.shape, i, subsample)
        orientation_histogram[:, :, i] = temp_filt[subsample]

    ''' Calculate the no-motion bin '''
    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[:, :, -1] = temp_filt[subsample]

    # now for each cell, compute the histogram
    hof_image = None

    if visualise:
        from skimage import draw

        radius = min(cx, cy) // 2 - 1
        hof_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations-1):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = int(radius * cos(float(o) / orientations * np.pi))
                    dy = int(radius * sin(float(o) / orientations * np.pi))
                    rr, cc = draw.bresenham(centre[0] - dy, centre[1] - dx,
                                            centre[0] + dy, centre[1] + dx)
                    hof_image[rr, cc] += orientation_histogram[y, x, o]

    """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (hog) descriptors.
    """

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y+by, x:x+bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)

    """
    The final step collects the hof descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if visualise:
        return normalised_blocks.ravel(), hof_image
    else:
        return normalised_blocks.ravel()
    
def getEmbeddings(videopath):
    
    # use_cuda = True
    # if use_cuda and torch.cuda.is_available():
    #   net.cuda()

    
    fps = 30
    cap = cv2.VideoCapture(videopath)
    
    embeddings = []
    nof = fps
    preprocess = ToTensor()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess(frame).unsqueeze(0)
        embds = my_model(frame)
        embds = np.ravel(embds.detach().numpy())
        embeddings.append(embds)
#     preprocess = ToTensor()
#     frame = preprocess(frame).unsqueeze(0)
#     embds = my_model(frame)
    embeddings = np.asarray(embeddings)
    embed_names = ['embed_'+str(k) for k in range(len(embeddings[0]))]
    tuple_embeddings=[]
    for i in range(len(embeddings)):
        tuple_embeddings.append(tuple(embeddings[i]))
    df=pd.DataFrame(tuple_embeddings,columns=embed_names)
    df['video_path'] = videopath
    return df

def get_frames_per(videopath):    
    import cv2
    cap = cv2.VideoCapture(videopath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def videoToHOGfeats(video_path):
    cap = cv2.VideoCapture(video_path)
    hog_feats = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64,64))
        feats = hogDescriptor(frame)
        hog_feats.append(np.ravel(feats))
    col_names = list('feat_' + str(i) for i in range(feats.shape[-2]))
    hog_feats = np.asarray(hog_feats)
    tuple_feats = list(map(lambda x: tuple(hog_feats[x]), range(len(hog_feats))))
    df = pd.DataFrame(tuple_feats, columns = col_names)
    df['video_path'] = video_path
    return df

def hogDescriptor(image):
    winSize = (64,64)
    blockSize = (32,32)
    blockStride = (16,16)
    cellSize = (16,16)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    #winStride = (16,16)
    #padding = (8,8)
    #locations = ((10,20),)
    return hog.compute(image)

class pre_processing:
    def remove_extra(df,bagsize = 128):
        """
            remove_extra(df,bagsize)
        """
        temp=df.reset_index(drop=False)
        temp.rename(index=int,columns={'index':'old_index'},inplace=True)
        real_index = temp.index
        old_index = temp.old_index
        indexes=[]
        prev=0
        for i in range(len(df)):
            if (old_index[i]%bagsize==0):
                if ((i-prev)==bagsize):
                    indexes.extend([k for k in range(prev,i)])
                prev=i 
        temp.drop(['old_index'],axis=1,inplace=True)
        return temp.loc[indexes]
            
def get_hof(videopath):
    cap = cv2.VideoCapture(videopath)
    feat_sets = []
    ret, frame_prev = cap.read()
    if not ret:
        return
    frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    frame_prev = cv2.resize(frame_prev, (120,160))
    while cap.isOpened():
      ret, frame_next = cap.read()
      if not ret:
        break
      frame_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
      frame_next = cv2.resize(frame_next, (120,160))
      feat_sets.append(hof(flow = getFlow(frame_prev, frame_next), orientations = 5, cells_per_block = (6,6), pixels_per_cell = (20,20)))
      frame_prev = frame_next.copy()
    feat_sets = np.asarray(feat_sets)
    col_names = list('feat_' + str(i) for i in range(feat_sets.shape[-1]))
    #hog_feats = np.asarray(hog_feats)
    tuple_feats = list(map(lambda x: tuple(feat_sets[x]), range(len(feat_sets))))
    df = pd.DataFrame(tuple_feats, columns = col_names)
    df['video_path'] = videopath
    return df

def emi_model_metrics(model_prefix, x_test, y_test_bag, y_test, extracted_dir):
    # Network parameters for our LSTM + FC Layer
    NUM_HIDDEN = 128
    NUM_TIMESTEPS = x_test.shape[-2]
    NUM_FEATS = x_test.shape[-1]
    FORGET_BIAS = 1.0
    NUM_OUTPUT = y_test.shape[-1]
    USE_DROPOUT = True
    KEEP_PROB = 0.75

    # For dataset API
    PREFETCH_NUM = 5
    BATCH_SIZE = 32

    tf.reset_default_graph()

    inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
    emiLSTM = EMI_BasicLSTM(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS,
                            forgetBias=FORGET_BIAS, useDropout=USE_DROPOUT)
    emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy')

    # Construct the graph
    g1 = tf.Graph()
    with g1.as_default():
        x_batch, y_batch = inputPipeline()
        y_cap = emiLSTM(x_batch)
        emiTrainer(y_cap, y_batch)

    with g1.as_default():
        emiDriver = EMI_Driver(inputPipeline, emiLSTM, emiTrainer)

    emiDriver.initializeSession(g1)
    tf.reset_default_graph()
    emiDriver.loadSavedGraphToNewSession(MODEL_PREFIX, 1004)
    
    start = time.time()
    predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                                   minProb=0.99, keep_prob=1.0)
    bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)

    end = time.time()
    
    metrics_dict = {'time_run': end - start,
                    'y_pred': bagPredictions,
                    'y_test': np.argmax(y_test_bag, axis = 1),
                    'classification_report': classification_report(bagPredictions, np.argmax(y_test_bag, axis = 1)),
                    'confusion_matrix': confusion_matrix(bagPredictions, np.argmax(y_test_bag, axis = 1))}
    
    ### Hook up the model size later
    import pickle as pkl
    import datetime
    now = datetime.datetime.now()
    filename = list((str(now.year),"-",str(now.month),"-",str(now.day),"|",str(now.hour),"-",str(now.minute)))
    filename = ''.join(filename)
    pkl.dump(metrics_dict, open(extracted_dir + "/lstm_dict_" + filename + ".pkl",mode='wb'))
    
    return metrics_dict
    
    