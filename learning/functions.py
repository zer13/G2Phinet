import os
import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import io

def totorch(x):
    return torch.tensor(x,dtype=torch.float)

def tonumpy(y):
    return y.detach().cpu().numpy()

def plot_contour(xt, xz, y_plot, ax, title):
    cs = ax.contour(xt, xz, y_plot)
    ax.clabel(cs, inline=1, fontsize=10)
    ax.set_title(title)
    ax.set_xlabel('xt')
    ax.set_ylabel('xz')
    return

def plot_contourf(xt, xz, y_plot, fig, ax, title):
    cs = ax.contourf(xt, xz, y_plot, 100)
    fig.colorbar(cs, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('xt')
    ax.set_ylabel('xz')
    return

def plot_1D(x,y_pred,y_true,ax,title):
    ax.plot(x,y_pred,'-b',label='pred')
    ax.plot(x,y_true,'-r',label='true')
    ax.set_title(title)
    ax.legend()
    return

def plot_all(num_samples,fig,axs,xt,xz,y_pred,y_true,latent_samples,gtp_samples):
    for isample in range(num_samples):
        yi_pred = y_pred[isample,:,:]
        yi_true = y_true[isample,:,:]
        y_diff = yi_pred-yi_true
        L2err = np.sqrt(np.mean(y_diff**2))
        latent = latent_samples[isample,:]
        gtp = gtp_samples[isample,:]
        # Prediction
        ax = axs[int((isample//4)*2),int((isample%4)*3)]
        title = 'Sample # {} Prediction (Error: {:.5f})'.format(isample+1,L2err)
        plot_contour(xt, xz, yi_pred, ax, title)
        # True
        ax = axs[int((isample//4)*2),int((isample%4)*3+1)]
        title = 'Sample # '+str(isample+1)+' True' + '\n Latent: ({:.2f},{:.2f})'.format(latent[0], latent[1])
        plot_contour(xt, xz, yi_true, ax, title)
        # Error
        ax = axs[int((isample//4)*2),int((isample%4)*3+2)]
        title = 'Gtp: {}'.format(np.argmax(gtp), np.exp(latent[0]))
        plot_contourf(xt, xz, y_diff, fig, ax, title)
        # Horizontal (first xz)
        ax = axs[int((isample//4)*2+1),int((isample%4)*3)]
        yi1_pred = yi_pred[0,:]
        yi1_true = yi_true[0,:]
        plot_1D(xt, yi1_pred, yi1_true, ax, 'xt direction')
        # Vertical (first xt)
        ax = axs[int((isample//4)*2+1),int((isample%4)*3+1)]
        yi1_pred = yi_pred[:,0]
        yi1_true = yi_true[:,0]
        plot_1D(xz, yi1_pred, yi1_true, ax, 'xz direction')
        # Diagonal (for #xt=#xz)
        ax = axs[int((isample//4)*2+1),int((isample%4)*3+2)]
        yi1_pred = np.diagonal(yi_pred)
        yi1_true = np.diagonal(yi_true)
        if (xz.shape[0]<xt.shape[0]):
            xx = xz
        else:
            xx = xt
        plot_1D(xx, yi1_pred, yi1_true, ax, 'diagonal direction')
    return