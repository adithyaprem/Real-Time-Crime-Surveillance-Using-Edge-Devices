import data_process.make_hof_half
from data_process.make_hof_half import get_processed_hof
import pandas as pd
import os,datetime
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pathlib
import pickle as pkl
from scripts.process_functions import get_frames_per
from scripts.process_functions import hof,getFlow
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest = 'data', default = '\path\to\data', 
                    help = 'Path to the trimmed videos')
parser.add_argument('--bagsize', default=128, type=int,
                    help='Bag size selected for the model')
parser.add_argument('--reduction-factor', default=1, type=int,
                    help='Factor by which the sampling rate has to be reduced')
parser.add_argument('--save-path', default='',
                   help = 'Path to save the features')

if __name__ == '__main__':
    args = parser.parse_args()
    hof_df = get_processed_hof(src_folder=args.data, bagsize=args.bagsize, reduction_factor=args.reduction_factor)
    hof_df.to_csv(os.path.join(args.save_path, 'feats.csv'))