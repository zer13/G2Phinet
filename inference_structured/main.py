import os
from glob import glob
from shutil import rmtree
import torch
import time
import math
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import sys
from scipy import io
import itertools
import argparse

from readData import *
from NN import *
from inference import *
from functions import *

def main(idseeds, seeds, num_types, num_latent, num_epochs, plot_itvl, savefigs, epochs, data_itvl):

    os.makedirs('./Inference',exist_ok=True)

    num_infer = 1
    infer_gtp_pred = np.zeros((5,num_infer,num_types,num_types))          # (number runs, number of inference during training, number of test cases in one run, prob. of five types)
    infer_latent = np.zeros((5,num_infer,num_types,num_latent))
    infer_numcorrect = np.zeros((5,num_infer))            # (number runs, number of inference during training)
    infer_loss = np.zeros((5,num_infer,num_types))
    
    data_all = {}

    for test_set in [1,2,3,4,5]:
        num_correct = 0
        for isample in range(num_types):
            is_correct, gtp_pred, latent, loss, data_save = inference_single('test',idseeds,test_set,isample,seeds,num_types,num_latent, num_epochs, plot_itvl, savefigs, epochs, data_itvl, data_loc)
            data_all[(test_set,isample)] = data_save
            infer_gtp_pred[test_set-1,0,isample,:] = gtp_pred
            infer_latent[test_set-1,0,isample,:] = latent
            if is_correct:
                num_correct+=1
            infer_loss[test_set-1,0,isample] = loss
        infer_numcorrect[test_set-1,0] = num_correct

    io.savemat('./Inference/result_{}.mat'.format(idseeds),{'infer_gtp_pred':infer_gtp_pred,'infer_numcorrect':infer_numcorrect,'infer_latent':infer_latent,'infer_loss':infer_loss})
    np.save('./Inference/SavedOutputs_{}.npy'.format(idseeds), data_all)

def check_existence(num_seeds_tot, epochs):
    iscomplete = True
    for seed in range(1,num_seeds_tot+1):
        for epoch in epochs:
            for test_set in range(1,6):
                if not os.path.exists('{}/Seed{}/Models/model_{}_It{}.pt'.format(data_loc,seed,test_set,epoch)):
                    iscomplete = False
                    print('{}/Seed{}/Models/model_{}_It{}.pt'.format(data_loc,seed,test_set,epoch))
    return iscomplete

# ##### If running a batch job is needed #####
# parser = argparse.ArgumentParser()
# parser.add_argument('code',metavar='K', type=int)
# args = parser.parse_args()
# code0 = args.code-1         # code: 1-240; code0: 0-239
# choice_numx = code0//80
# idseeds = code0-choice_numx*80
# if choice_numx == 0:
#     data_itvl = 6
# elif choice_numx == 1:
#     data_itvl = 10
# else:
#     data_itvl = 15
# choices = []
# num_seeds_tot = 67

# for i in range(20):
#     choices+=[[i+1]]

# for i in range(20):
#     choices+=[list(range(3*i+1,3*i+4))]

# for i in range(20):
#     choices+=[list(range(3*i+1,3*i+7))]

# for i in range(20):
#     choices+=[list(range(3*i+1,3*i+11))]

seeds = range(1,68)
idseeds = 0             # an id for the current choice of seeds
epochs = [16000, 18000, 20000]
data_itvl = 15

num_latent = 2
num_types = 4

num_epochs = 1000
plot_itvl = 1000
savefigs = True

data_loc = '../learning'

main(idseeds, seeds, num_types, num_latent, num_epochs, plot_itvl, savefigs, epochs, data_itvl)