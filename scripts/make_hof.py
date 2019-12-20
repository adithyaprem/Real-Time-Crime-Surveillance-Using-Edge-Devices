from video_stuff import get_frames_per
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy.stats import iqr
import pandas as pd 
from video_stuff import get_hof,pre_processing
import os

def visualize(src_folder, f = True):    
    """
    
    src_folder path must contain only the collection of  class folders that contains corresponding data
    
    """
    dest=os.listdir(src_folder)
    fil=[]
    if f:
        for i in dest:
            for j in os.listdir(src_folder+'/'+i):
                fil.append(src_folder+'/'+i+'/'+j)
    else:
        for i in dest:
            fil.append(src_folder+'/'+i)
    frames=[]
    for file in fil:
        frames.append(get_frames_per(file))
    plt.boxplot(frames)
    plt.ylabel("Frames in a Video")
    frames_df=pd.DataFrame(frames)
    print("Total no of frames : ")
    print(frames_df.sum())
    print(frames_df.describe())
   
    return frames_df



def get_processed_hof(src_folder,bagsize):
    """
        get_processed_hof(src_folder,bagsize)
        
        returns df with extra removed based on bagsize
    
    """
    
    dest=os.listdir(src_folder)
    dfa=pd.DataFrame()
    counts=0
    for i in dest:
        if len(os.listdir(os.path.join(src_folder, i))) > 0:
            for j in os.listdir(os.path.join(src_folder, i)):
                df=get_hof(os.path.join(src_folder,i,j))
                if df is None:
                    continue
                df['label']=counts
                dfa=pd.concat([dfa,df])
            counts=counts+1
    
    dd=pre_processing.remove_extra(dfa,bagsize)
    return dd