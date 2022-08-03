import os
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
import argparse

from readData import *
from NN import *
from functions import *

def main(num_types,seed,test_set,savefigs):

    os.makedirs('./Models/Seed{}'.format(seed),exist_ok=True)
    os.makedirs('./Outputs/Seed{}'.format(seed),exist_ok=True)
    if savefigs:
        os.makedirs('./Fig/Seed{}/test{}'.format(seed,test_set),exist_ok=True)
        os.makedirs('./Fig/Seed{}/test{}/Train'.format(seed,test_set),exist_ok=True)
        os.makedirs('./Fig/Seed{}/test{}/Test'.format(seed,test_set),exist_ok=True)
        os.makedirs('./Fig/Seed{}/test{}/Loss'.format(seed,test_set),exist_ok=True)
        os.makedirs('./Fig/Seed{}/test{}/Latent'.format(seed,test_set),exist_ok=True)
        os.makedirs('./Fig/Seed{}/test{}/Interp'.format(seed,test_set),exist_ok=True)

    # Import Data
    xt, xz, y_train, y_trainplot, y_test, gtp_train, gtp_trainplot, gtp_test, num_cases, num_training, num_trainplot, num_test = import_data(num_types, test_set)

    w_train = alpha_R*np.ones((y_train.shape[0],1))
    w_trainplot = alpha_R*np.ones((y_trainplot.shape[0],1))
    w_test = alpha_R*np.ones((y_test.shape[0],1))

    ## initialization          
    model = G2Phi(xz, xt, num_latent, num_types, alphas)

    loss_train = []
    loss_test = []

    saved = {}

    ### training and testing ###
    for iepoch in range(num_epochs):
        ### Training ###
        tic1 = time.time()
        #print('lr:'+str(model.optimizer.param_groups[0]['lr']))
        if(iepoch%1==0):
            y_train_syn0, gtp_train_syn0, w_train_syn0 = oversampling(y_train, gtp_train, alpha_R)
            y_train_bal = np.concatenate((y_train,y_train_syn0),axis=0)
            gtp_train_bal = np.concatenate((gtp_train,gtp_train_syn0),axis=0)

            y_train_syn1, gtp_train_syn1, w_train_syn1 = mixup(y_train_bal, gtp_train_bal, num_mixup, alpha_M, num_types)

            y_train_all = np.concatenate((y_train_bal,y_train_syn1),axis=0)
            gtp_train_all = np.concatenate((gtp_train_bal,gtp_train_syn1),axis=0)
            w_train_all = np.concatenate((w_train,w_train_syn0,w_train_syn1),axis=0)
        y_pred_train_all, latent_train_all = model(totorch(y_train_all), totorch(gtp_train_all))
        loss, loss_R, loss_G = model.losses(totorch(y_train_all), y_pred_train_all, latent_train_all, totorch(w_train_all))
        loss_train.append([tonumpy(loss),tonumpy(loss_R),tonumpy(loss_G)])
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        #if (iepoch>=0):
        model.scheduler.step()
        tic2 = time.time()
        print('Epoch: {}, Loss (tot,R,G): {:.4f}, {:.4f}, {:.4f}, Times: {:.2f}'.format(
                iepoch+1, tonumpy(loss),tonumpy(loss_R),tonumpy(loss_G), tic2-tic1),
                flush=True)

        ### Test ###
        if ((iepoch+1)%test_itvl == 0):
            ###### Test on Training Data ######
            y_pred_trainplot, latent_trainplot = model(totorch(y_trainplot), totorch(gtp_trainplot))
            y_pred_trainplot = tonumpy(y_pred_trainplot)
            latent_trainplot = tonumpy(latent_trainplot)
            ###### Test ######
            y_pred_test, latent_test = model(totorch(y_test), totorch(gtp_test))
            loss, loss_R, loss_G = model.losses(totorch(y_test), y_pred_test, latent_test, totorch(w_test))
            loss_test.append([tonumpy(loss),tonumpy(loss_R),tonumpy(loss_G)])
            y_pred_test = tonumpy(y_pred_test)
            latent_test = tonumpy(latent_test)

        ### Plot ###
        if ((iepoch+1)%plot_itvl == 0):

            ###### Test on Training Data ssaa ######
            fig, axs = plt.subplots(nrows = np.int(np.ceil(num_trainplot/4)*2), ncols = 12, figsize = (38.4,np.int(np.ceil(num_trainplot/4))*3.0*2))
            plot_all(num_trainplot,fig,axs,xt,xz,y_pred_trainplot[:,0],y_trainplot[:,0],latent_trainplot,gtp_trainplot)
            L2err_tot = np.sqrt(np.mean((y_pred_trainplot[:,0]-y_trainplot[:,0])**2))
            fig.suptitle('Train (Epoch {}) L2: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if (savefigs):
                plt.savefig('./Fig/Seed{}/test{}/Train/Fig_Train_saa_epoch{}.png'.format(seed,test_set,iepoch+1))
            plt.close()
            # scc
            fig, axs = plt.subplots(nrows = np.int(np.ceil(num_trainplot/4)*2), ncols = 12, figsize = (38.4,np.int(np.ceil(num_trainplot/4))*3.0*2))
            plot_all(num_trainplot,fig,axs,xt,xz,y_pred_trainplot[:,1],y_trainplot[:,1],latent_trainplot,gtp_trainplot)
            L2err_tot = np.sqrt(np.mean((y_pred_trainplot[:,1]-y_trainplot[:,1])**2))
            fig.suptitle('Train (Epoch {}) L2: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if (savefigs):
                plt.savefig('./Fig/Seed{}/test{}/Train/Fig_Train_scc_epoch{}.png'.format(seed,test_set,iepoch+1))
            plt.close()
            ###### Test saa ######
            fig, axs = plt.subplots(nrows = np.int(np.ceil(num_test/4)*2), ncols = 12, figsize = (38.4,np.int(np.ceil(num_test/4))*3.0*2))
            plot_all(num_test,fig,axs,xt,xz,y_pred_test[:,0],y_test[:,0],latent_test,gtp_test)
            L2err_tot = np.mean((y_pred_test[:,0]-y_test[:,0])**2)
            fig.suptitle('Test (Epoch {}) L2: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if (savefigs):
                plt.savefig('./Fig/Seed{}/test{}/Test/Fig_Test_saa_epoch{}.png'.format(seed,test_set,iepoch+1))
            plt.close()
            # scc
            fig, axs = plt.subplots(nrows = np.int(np.ceil(num_test/4)*2), ncols = 12, figsize = (38.4,np.int(np.ceil(num_test/4))*3.0*2))
            plot_all(num_test,fig,axs,xt,xz,y_pred_test[:,1],y_test[:,1],latent_test,gtp_test)
            L2err_tot = np.mean((y_pred_test[:,1]-y_test[:,1])**2)
            fig.suptitle('Test (Epoch {}) L2: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if (savefigs):
                plt.savefig('./Fig/Seed{}/test{}/Test/Fig_Test_scc_epoch{}.png'.format(seed,test_set,iepoch+1))
            plt.close()
            ###### Loss ######
            plt.figure(111)
            plt.semilogy(np.arange(iepoch+1),np.array(loss_train)[:,0],'-b', label='Train')
            plt.semilogy((np.arange((iepoch+1)//test_itvl)+1)*test_itvl,np.array(loss_test)[:,0],'-r', label = 'Test')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            if (savefigs):
                plt.savefig('./Fig/Seed{}/test{}/Loss/Fig_loss_epoch{}.png'.format(seed,test_set,iepoch+1))
            plt.close(111)

            ###### Latent Distribution ######
            # Only plot the two unsupervised
            fig, ax = plt.subplots(1,1,figsize=(6.4,4.8))
            num_bal = y_train_bal.shape[0]
            num_syn1 = y_train_syn1.shape[0]
            #num_syn2 = y_train_syn2.shape[0]
            ax.plot(tonumpy(latent_train_all[:num_training,0]), tonumpy(latent_train_all[:num_training,1]), 'xb', label = 'train')
            ax.plot(tonumpy(latent_train_all[num_training:num_bal,0]), tonumpy(latent_train_all[num_training:num_bal,1]), '+b', label = 'train (oversampled)')
            ax.plot(tonumpy(latent_train_all[num_bal:num_bal+num_syn1,0]), tonumpy(latent_train_all[num_bal:num_bal+num_syn1,1]), '^b', label = 'train (mixup)')
            #ax.plot(tonumpy(latent_train_all[num_bal+num_syn1:num_bal+num_syn1+num_syn2,0]), tonumpy(latent_train_all[num_bal+num_syn1:num_bal+num_syn1+num_syn2,1]), 'xk', label = 'train (wrong categories)')
            ax.plot(latent_test[:,0], latent_test[:,1], '.r', label = 'test')
            ax.set_xlabel('Latent var. #1')
            ax.set_ylabel('Latent var. #2')
            ax.legend()
            if (savefigs):
                plt.savefig('./Fig/Seed{}/test{}/Latent/Fig_lat12_epoch{}.png'.format(seed,test_set,iepoch+1))
            plt.close()

        if ((iepoch+1)%save_itvl == 0):
            saved[iepoch+1] = (xt, xz, y_pred_trainplot, y_trainplot, latent_trainplot, gtp_trainplot, y_pred_test, y_test, latent_test, gtp_test, loss_train, loss_test, latent_train_all, latent_test, num_bal, num_syn1)

            np.save('./Outputs/Seed{}/SavedOutputs_{}_It{}.npy'.format(seed,test_set, iepoch+1), saved)

            torch.save(model, './Models/Seed{}/model_{}_It{}.pt'.format(seed,test_set,iepoch+1))

##### If running a batch job is needed #####
# parser = argparse.ArgumentParser()
# parser.add_argument('code',metavar='K', type=int)
# args = parser.parse_args()
# ### code = seed*5+(testset-1) ###
# seed = (args.code-1)//5+1
# testset = args.code-((args.code-1)//5)*5
# print('========Seed and testset: ({}, {})========'.format(seed,testset))

np.set_printoptions(threshold=sys.maxsize)

testset_all = [1,2,3,4,5]         # 5-fold test data; in {1,2,3,4,5}
seed_all = range(1,68)             # choice of random seeds

num_latent = 2      # d_\eta

num_types = 4       # Number of genotypes (material classes)

num_mixup = 56      # number of mixup cases; here 56=2*28, twice the number of the original samples

# Loss weights
alpha_R = 1.0   # Stress Reconstruction
alpha_M = 0.01   # Weights for synthetic mixup data
alpha_G = 0.01  # Centered Unsupervised Latent Space
alphas = (alpha_R, alpha_M, alpha_G)

# Number of training epochs
num_epochs = 20000

# Test/Plot/Save per X epochs
test_itvl = 100
save_itvl = 2000
plot_itvl = 2000

# Save the figures or not?
savefigs = True

for seed in seed_all:
    for testset in testset_all:

        np.random.seed(seed)
        torch.manual_seed(seed)

        main(num_types,seed,testset,savefigs)