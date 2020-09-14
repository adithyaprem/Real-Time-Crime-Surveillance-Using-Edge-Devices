def emirnn_preprocess(bagsize,no_of_features,extractedDir,numClass,subinstanceLen, subinstanceStride,data_csv,raw_create):
    
    import pandas as pd
    import numpy as np
    import os 
    import video_stuff
    numSteps=bagsize
    numFeats=no_of_features
    if type(data_csv)=='str':
        dataset_name = pd.read_csv(data_csv,index_col=0)
        labels = dataset_name['label']
    else:
        labels=data_csv['label']
        dataset_name=data_csv


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
        if (os.path.isdir(extractedDir+'/'+'RAW')):
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

    def makeEMIData(subinstanceLen, subinstanceStride, sourceDir, outDir,numClass,numSteps,numFeats):
        x_train, y_train, x_test, y_test, x_val, y_val = loadData(sourceDir)
        x, y = bagData(x_train, y_train, subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
        np.save(outDir + '/x_train.npy', x)
        np.save(outDir + '/y_train.npy', y)
        print('Num train %d' % len(x))
        x, y = bagData(x_test, y_test, subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
        np.save(outDir + '/x_test.npy', x)
        np.save(outDir + '/y_test.npy', y)
        print('Num test %d' % len(x))
        x, y = bagData(x_val, y_val, subinstanceLen, subinstanceStride,numClass,numSteps,numFeats)
        np.save(outDir + '/x_val.npy', x)
        np.save(outDir + '/y_val.npy', y)
        print('Num val %d' % len(x))



                                                                                        #subinstanceLen = 12
                                                                                        #subinstanceStride = 3
                                                                                        #extractedDir = '/home/adithyapa4444_gmail_com'
    rawDir = extractedDir + '/RAW'
    sourceDir = rawDir
    from os import mkdir
    # WHEN YOU CHANGE THE ABOVE - CREATE A FOLDER 
    if (not(os.path.isdir(extractedDir+'/'+str(subinstanceLen)+'_'+str(subinstanceStride)))):
        mkdir(extractedDir+'/'+str(subinstanceLen)+'_'+str(subinstanceStride))  

    outDir = extractedDir + '%d_%d' % (subinstanceLen, subinstanceStride)
    makeEMIData(subinstanceLen, subinstanceStride, sourceDir, outDir,numClass,numSteps,numFeats)
    
    print('preprocessing is over....')
    return


