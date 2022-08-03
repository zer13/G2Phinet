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

from readData import *
from NN import *
from NN2 import *
from functions import *

def inference_single(which, idseeds, test_set, id, seeds, num_types, num_latent, num_epochs, plot_itvl, savefigs, epochs, data_itvl, data_loc):

    foldername = 'test{}-{}-{}'.format(idseeds,test_set,id)
    if savefigs:
        os.makedirs('Inference/{}'.format(foldername),exist_ok=True)
    
    print('==========Fitting for a Chosen Sample: {}=========='.format(foldername))

    xt, xz, y_train, y_trainplot, y_test, gtp_train, gtp_trainplot, gtp_test, num_cases, num_training, num_trainplot, num_test = import_data(num_types, test_set)
    model_DON_list = []

    for seed in seeds:
        for i0epoch in epochs:
            model_DON = torch.load('{}/Models/Seed{}/model_{}_It{}.pt'.format(data_loc,seed,test_set,i0epoch))
            model_DON_list.append(model_DON)
    
    print('Num. Models in the Megamodel: {}'.format(len(model_DON_list)))
    model = G2Phi_Inference(model_DON_list,num_types,num_latent)

    xt_i = xt[::data_itvl]
    xz_i = xz[::data_itvl]

    if (which == 'train'):
        y_i = y_trainplot[id:id+1,:,::data_itvl,::data_itvl]
        gtp_true = gtp_trainplot[id]
    elif (which == 'test'):
        y_i = y_test[id:id+1,:,::data_itvl,::data_itvl]
        gtp_true = gtp_test[id]

    losses = []
    gtps_pred = []
    its = []

    gtp_pred = tonumpy(model.label[0,:])
    gtps_pred.append(gtp_pred)
    its.append(0)

    if (which == 'train'):
        y_true_all = y_trainplot[id]
    elif (which == 'test'):
        y_true_all = y_test[id]

    data_save = {}

    for iepoch in range(num_epochs):
        ### Training ###
        tic = time.time()
        y_i_pred_list, y_i_pred_mean = model(totorch(xz_i),totorch(xt_i))
        loss,loss_R,loss_G = model.loss(y_i_pred_list,totorch(y_i))
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
            print('Epoch: {}, Losses (tot,R,G): {:.4f},{:.4f},{:.4f}, Time: {:.2f}'.format(iepoch+1, loss, loss_R, loss_G,toc-tic),flush=True)
            print('Label: {}, First Latent: {}'.format(tonumpy(model.label[0]),tonumpy(model.latent[0])))
            print('True gtp: {}'.format(gtp_true))
            ### Testing ###
            gtp_pred = tonumpy(model.label[0,:])
            gtps_pred.append(gtp_pred)
            its.append(iepoch+1)
            _, y_pred = model(totorch(xz),totorch(xt))
            y_pred_all = tonumpy(y_pred)[0]

            L2err = np.sqrt(np.mean((y_pred_all-y_true_all)**2)/np.mean(y_true_all**2))

        ### Plot ###
        if ((iepoch+1)%plot_itvl == 0):

            # print(list(model.parameters())[0])
            # print(list(model.parameters())[1])

            # print(np.shape(thetas_pred))
            # print(thetas_pred)
            # print(its)

            ############### saa ###############

            fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (9.6,4.8))
            Xt_i,Xz_i=np.meshgrid(xt_i,xz_i)
            Xt_i = Xt_i.reshape([-1,1])
            Xz_i = Xz_i.reshape([-1,1])

            y_true = y_true_all[0]
            y_pred = y_pred_all[0]

            plot_contour(xt,xz,y_pred,axs[0,0],'pred')
            axs[0,0].plot(Xt_i,Xz_i,'xk')
            plot_contour(xt,xz,y_true,axs[0,1],'true')
            axs[0,1].plot(Xt_i,Xz_i,'xk')
            plot_contourf(xt,xz,y_pred-y_true,fig,axs[0,2],'diff')
            axs[0,2].plot(Xt_i,Xz_i,'xk')

            # Horizontal (first xz)
            y1_pred = y_pred[0,:]
            y1_true = y_true[0,:]
            if (gtp_true.shape[0] == 4):
                title = 'Gtp_true: ({:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_true[0]), (gtp_true[1]), (gtp_true[2]), (gtp_true[3]))
            else:
                title = 'Gtp_true: ({:.2f},{:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_true[0]), (gtp_true[1]), (gtp_true[2]), (gtp_true[3]), (gtp_true[4]))
            plot_1D(xt, y1_pred, y1_true, axs[1,0], 'xt direction \n' + title, plot_data=True, data_itvl=data_itvl)
                    
            # Vertical (first xt)
            y1_pred = y_pred[:,0]
            y1_true = y_true[:,0]
            if (gtp_true.shape[0] == 4):
                title = 'Gtp_pred: ({:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_pred[0]), (gtp_pred[1]), (gtp_pred[2]), (gtp_pred[3]))
            else:
                title = 'Gtp_pred: ({:.2f},{:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_pred[0]), (gtp_pred[1]), (gtp_pred[2]), (gtp_pred[3]), (gtp_pred[4]))
            plot_1D(xz, y1_pred, y1_true, axs[1,1], 'xz direction \n' + title, plot_data=True, data_itvl=data_itvl)

            # Diagonal (for #xt=#xz)
            y1_pred = np.diagonal(y_pred)
            y1_true = np.diagonal(y_true)
            if (xz.shape[0]<xt.shape[0]):
                xx = xz
            else:
                xx = xt
            #title = 'Latent: ({:.2f},{:.2f})'.format(model.latent[0,0],model.latent[0,1])
            plot_1D(xx, y1_pred, y1_true, axs[1,2], 'diagonal direction \n' + title, plot_data=True, data_itvl=data_itvl)

            L2err_tot = np.sqrt(np.mean((y_pred-y_true)**2))
            fig.suptitle('Result (Epoch {}) L2: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if savefigs:
                plt.savefig('./Inference/{}/Curve_Saa_{}'.format(foldername,iepoch+1))
            plt.close()

            info_saa = (xt,xz,y_pred,y_true,Xt_i,Xz_i)


            ############### scc ###############

            fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (9.6,4.8))
            Xt_i,Xz_i=np.meshgrid(xt_i,xz_i)
            Xt_i = Xt_i.reshape([-1,1])
            Xz_i = Xz_i.reshape([-1,1])

            y_true = y_true_all[1]
            y_pred = y_pred_all[1]

            plot_contour(xt,xz,y_pred,axs[0,0],'pred')
            axs[0,0].plot(Xt_i,Xz_i,'xk')
            plot_contour(xt,xz,y_true,axs[0,1],'true')
            axs[0,1].plot(Xt_i,Xz_i,'xk')
            plot_contourf(xt,xz,y_pred-y_true,fig,axs[0,2],'diff')
            axs[0,2].plot(Xt_i,Xz_i,'xk')

            # Horizontal (first xz)
            y1_pred = y_pred[0,:]
            y1_true = y_true[0,:]
            if (gtp_true.shape[0] == 4):
                title = 'Gtp_true: ({:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_true[0]), (gtp_true[1]), (gtp_true[2]), (gtp_true[3]))
            else:
                title = 'Gtp_true: ({:.2f},{:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_true[0]), (gtp_true[1]), (gtp_true[2]), (gtp_true[3]), (gtp_true[4]))
            plot_1D(xt, y1_pred, y1_true, axs[1,0], 'xt direction \n' + title, plot_data=True, data_itvl=data_itvl)
                    
            # Vertical (first xt)
            y1_pred = y_pred[:,0]
            y1_true = y_true[:,0]
            if (gtp_true.shape[0] == 4):
                title = 'Gtp_pred: ({:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_pred[0]), (gtp_pred[1]), (gtp_pred[2]), (gtp_pred[3]))
            else:
                title = 'Gtp_pred: ({:.2f},{:.2f},{:.2f},{:.2f},{:.2f})'.format((gtp_pred[0]), (gtp_pred[1]), (gtp_pred[2]), (gtp_pred[3]), (gtp_pred[4]))
            plot_1D(xz, y1_pred, y1_true, axs[1,1], 'xz direction \n' + title, plot_data=True, data_itvl=data_itvl)

            # Diagonal (for #xt=#xz)
            y1_pred = np.diagonal(y_pred)
            y1_true = np.diagonal(y_true)
            if (xz.shape[0]<xt.shape[0]):
                xx = xz
            else:
                xx = xt
            #title = 'Latent: ({:.2f},{:.2f})'.format(model.latent[0,0],model.latent[0,1])
            plot_1D(xx, y1_pred, y1_true, axs[1,2], 'diagonal direction \n' + title, plot_data=True, data_itvl=data_itvl)

            L2err_tot = np.sqrt(np.mean((y_pred-y_true)**2))
            fig.suptitle('Result (Epoch {}) L2: {}'.format(iepoch+1,L2err_tot))
            plt.tight_layout()
            if savefigs:
                plt.savefig('./Inference/{}/Curve_Scc_{}'.format(foldername,iepoch+1))
            plt.close()

            info_scc = (xt,xz,y_pred,y_true,Xt_i,Xz_i)


            plt.figure(2)
            plt.semilogy(np.arange(iepoch+1)+1,np.array(losses),'-b')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.tight_layout()
            if savefigs:
                plt.savefig('./Inference/{}/Loss_{}'.format(foldername,iepoch+1))
            plt.close()

            info_loss = (np.arange(iepoch+1)+1,np.array(losses))

            plt.figure(3)
            gtp_true_id = np.argmax(gtp_true)
            plt.plot(np.array(its),np.array(gtps_pred)[:,gtp_true_id],'-b',label='Prediction')
            plt.plot(np.array(its),np.ones_like(np.array(gtps_pred)[:,gtp_true_id]),'--r',label='True')
            plt.title('Estimated and True Probability')
            plt.xlabel('Epoch')
            plt.ylabel('Prob. Value')
            plt.tight_layout()
            if savefigs:
                plt.savefig('./Inference/{}/Param_{}'.format(foldername,iepoch+1))
            plt.close()

            info_prob = (np.array(its),np.array(gtps_pred)[:,gtp_true_id])

            info = (info_saa, info_scc, info_loss, info_prob)
            data_save[iepoch+1] = info

    true_id = np.argmax(gtp_true)
    pred_id = np.argmax(gtp_pred)
    return true_id==pred_id,gtp_pred,0.014,L2err,data_save