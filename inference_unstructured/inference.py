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

from readData2 import *
from readData import *
from NN import *
from NN2 import *
from functions import *

def plot_curves(ax, x_i, y_i, x_true, y_true, x_pred, y_pred, c, testname):
    ax.plot(x_i, g(y_i),'.'+c,label=testname+' data')
    ax.plot(x_true,g(y_true),'x'+c,label=testname+' true')
    ax.plot(x_pred,g(y_pred),'^'+c,label=testname+' pred')
    return

def g(x):
    return x #np.exp(5*x)-1


def inference_single(idseeds, test_set, id, seeds, num_types, num_latent, num_epochs, plot_itvl, savefigs, epochs, choice_numx, data_loc):

    foldername = 'test{}-{}-{}'.format(idseeds,test_set,id)
    if savefigs:
        os.makedirs('Inference/{}'.format(foldername),exist_ok=True)

    print('==========Fitting for a Chosen Sample: {}=========='.format(foldername))

    xz, xt, yz, yt, is_data, gtp_true, is_type_i = import_data2(num_types, test_set, id, choice_numx)
    xz_i = xz[is_data]      # 1d
    xt_i = xt[is_data]      # 1d
    y = np.concatenate((yz, yt),axis=1)     # (?, 2)
    y_i = y[is_data]                        # (??, 2)
    y_i_torch = totorch(y_i.T[None,:])
    y_torch = totorch(y.T[None,:])      # (1, 2, ??)

    model_DON_list = []

    for seed in seeds:
        for i0epoch in epochs:
            model_DON = torch.load('{}/Models/Seed{}/model_{}_It{}.pt'.format(data_loc,seed,test_set,i0epoch))
            model_DON_list.append(model_DON)

    print('Num. Models in the Megamodel: {}'.format(len(model_DON_list)))
    model = G2Phi_Inference(model_DON_list,num_types,num_latent)

    losses = []
    gtps_pred = []
    its = []

    gtp_pred = tonumpy(model.label[0,:])
    gtps_pred.append(gtp_pred)
    its.append(0)

    data_save = {}

    for iepoch in range(num_epochs):
        ### Training ###
        tic = time.time()
        y_i_pred_list, y_i_pred_mean = model(totorch(xz_i),totorch(xt_i),False)
        loss,loss_R,loss_G = model.loss(y_i_pred_list,y_i_torch)
        model.optimizer.zero_grad()
        #loss.backward(retain_graph=True)
        loss.backward()
        model.optimizer.step()
        loss = tonumpy(loss)
        loss_R = tonumpy(loss_R)
        loss_G = tonumpy(loss_G)
        losses.append(loss)
        toc = time.time()
        if ((iepoch+1)%100==0):

            xt_, xz_, _, _, y_test_, _, _, gtp_test_, _, _, _, _ = import_data(num_types, test_set)
            gtp_true_ = gtp_test_[id]
            y_true_ = y_test_[id]

            print('Epoch: {}, Losses (tot,R,G): {:.4f},{:.4f},{:.4f}, Time: {:.2f}'.format(iepoch+1, loss, loss_R, loss_G,toc-tic),flush=True)
            print('Label: {}, First Latent: {}'.format(tonumpy(model.label[0]),tonumpy(model.latent[0])))
            print('True gtp: {}'.format(gtp_true))
            ### Testing ###
            gtp_pred = tonumpy(model.label[0,:])
            gtps_pred.append(gtp_pred)
            its.append(iepoch+1)
            _, y_pred_ = model(totorch(xz_),totorch(xt_),True)
            y_pred_ = tonumpy(y_pred_)[0]

            L2err = np.sqrt(np.mean((y_pred_-y_true_)**2)/np.mean(y_true_**2))

        ### Plot ###
        if ((iepoch+1)%plot_itvl == 0):


            ############### saa ###############

            fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (9.6,3.2))

            plot_contour(xt_,xz_,y_pred_[0],axs[0],'pred')
            axs[0].plot(xt_i,xz_i,'xk')
            axs[0].set_xlim([0,0.65])
            axs[0].set_ylim([0,0.65])
            plot_contour(xt_,xz_,y_true_[0],axs[1],'true')
            axs[1].plot(xt_i,xz_i,'xk')
            axs[1].set_xlim([0,0.65])
            axs[1].set_ylim([0,0.65])
            plot_contourf(xt_,xz_,y_pred_[0]-y_true_[0],fig,axs[2],'diff')
            axs[2].plot(xt_i,xz_i,'xk')
            axs[2].set_xlim([0,0.65])
            axs[2].set_ylim([0,0.65])

            L2err_tot = np.sqrt(np.mean((y_pred_[0]-y_true_[0])**2))/np.sqrt(np.mean((y_true_[0])**2))
            fig.suptitle('Result (Epoch {}) L2 Rel: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if savefigs:
                plt.savefig('./Inference/{}/Curve_Saa_{}'.format(foldername,iepoch+1))
            plt.close()

            info_saa = (xt_,xz_,y_pred_[0],y_true_[0],xt_i,xz_i)

            ############### scc ###############

            fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (9.6,3.2))

            plot_contour(xt_,xz_,y_pred_[1],axs[0],'pred')
            axs[0].plot(xt_i,xz_i,'xk')
            axs[0].set_xlim([0,0.65])
            axs[0].set_ylim([0,0.65])
            plot_contour(xt_,xz_,y_true_[1],axs[1],'true')
            axs[1].plot(xt_i,xz_i,'xk')
            axs[1].set_xlim([0,0.65])
            axs[1].set_ylim([0,0.65])
            plot_contourf(xt_,xz_,y_pred_[1]-y_true_[1],fig,axs[2],'diff')
            axs[2].plot(xt_i,xz_i,'xk')
            axs[2].set_xlim([0,0.65])
            axs[2].set_ylim([0,0.65])

            L2err_tot = np.sqrt(np.mean((y_pred_[1]-y_true_[1])**2))/np.sqrt(np.mean((y_true_[1])**2))
            fig.suptitle('Result (Epoch {}) L2 Rel: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if savefigs:
                plt.savefig('./Inference/{}/Curve_Scc_{}'.format(foldername,iepoch+1))
            plt.close()

            info_scc = (xt_,xz_,y_pred_[1],y_true_[1],xt_i,xz_i)

            info = (info_saa, info_scc)
            data_save[iepoch+1] = info

    true_id = np.argmax(gtp_true)
    pred_id = np.argmax(gtp_pred)
    return true_id==pred_id,gtp_pred,0.014,L2err,data_save