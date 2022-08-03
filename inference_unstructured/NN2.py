import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class G2Phi_Inference(nn.Module):
    def __init__(self, model_DON_list, num_types, num_latent):
        super(G2Phi_Inference, self).__init__()

        self.model_DON_list = model_DON_list
        self.num_models = len(model_DON_list)

        self.softmax = nn.Softmax(dim=1)

        for param in model_DON_list[0].parameters():
            param.requires_grad = False

        # latent not shared
        self.latent = nn.Parameter(torch.tensor(np.zeros((self.num_models,num_latent)),dtype=torch.float,requires_grad=True))

        # label shared
        self.label_logits = nn.Parameter(torch.tensor(np.zeros((1,num_types)),dtype=torch.float,requires_grad=True))

        self.label = self.softmax(self.label_logits)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.1)

    def forward(self, xz, xt, is_structured):
        if (is_structured):
            return self._forward_structured(xz, xt)
        else:
            return self._forward_unstructured(xz, xt)

    def _forward_structured(self, xz, xt):
        # xz, xt: (?,)

        # label shared
        self.label = self.softmax(self.label_logits)

        # those related to latent not shared
        y_pred_list = []
        is_first = True
        for imodel, model_DON in enumerate(self.model_DON_list):
            b = model_DON.forward_decoder(self.latent[imodel:imodel+1,:],self.label)

            Xt, Xz = np.meshgrid(xt, xz)
            X = np.concatenate((Xt.reshape(-1,1),Xz.reshape(-1,1)),axis=1)
            X = torch.tensor(X, dtype=torch.float, requires_grad=True)      # size_batch * 2; columns: Xz, Xt

            t = model_DON.trunk(X)

            size_batch = b.shape[0]
            num_x = t.shape[0]
            num_points_z = xz.shape[0]
            num_points_t = xt.shape[0]

            b1 = b[:,:model_DON.p]
            b2 = b[:,model_DON.p:]
            t1 = t[:,:model_DON.p]
            t2 = t[:,model_DON.p:]

            temp1 = (torch.matmul(b1,t1.T)+model_DON.bias[0])*(X[:,0:1]**2+X[:,1:2]**2).T    # (size_batch, num_x)
            temp2 = (torch.matmul(b2,t2.T)+model_DON.bias[1])*(X[:,0:1]**2+X[:,1:2]**2).T    # (size_batch, num_x)
            y1_pred = temp1.reshape([size_batch,1,num_points_z,num_points_t])
            y2_pred = temp2.reshape([size_batch,1,num_points_z,num_points_t])
            y_pred = torch.cat((y1_pred,y2_pred),dim=1)       # (size_batch, 2, num_z, num_t)

            y_pred_list.append(y_pred)
            if is_first:
                y_pred_sum = y_pred
            else:
                y_pred_sum = y_pred_sum + y_pred
            is_first = False
        y_pred_mean = y_pred_sum / self.num_models
        return y_pred_list, y_pred_mean

    def _forward_unstructured(self, xz, xt):
        # xz, xt: (?,)

        # label shared
        self.label = self.softmax(self.label_logits)

        # those related to latent not shared
        y_pred_list = []
        is_first = True
        for imodel, model_DON in enumerate(self.model_DON_list):
            b = model_DON.forward_decoder(self.latent[imodel:imodel+1,:],self.label)

            X = np.concatenate((xt[:,None],xz[:,None]),axis=1)
            X = torch.tensor(X, dtype=torch.float, requires_grad=True)      # size_batch * 2; columns: Xz, Xt

            t = model_DON.trunk(X)

            size_batch = b.shape[0]
            num_x = t.shape[0]

            b1 = b[:,:model_DON.p]
            b2 = b[:,model_DON.p:]
            t1 = t[:,:model_DON.p]
            t2 = t[:,model_DON.p:]

            temp1 = (torch.matmul(b1,t1.T)+model_DON.bias[0])*(X[:,0:1]**2+X[:,1:2]**2).T    # (size_batch, num_x)
            temp2 = (torch.matmul(b2,t2.T)+model_DON.bias[1])*(X[:,0:1]**2+X[:,1:2]**2).T    # (size_batch, num_x)
            y1_pred = temp1.reshape([size_batch,1,num_x])
            y2_pred = temp2.reshape([size_batch,1,num_x])
            y_pred = torch.cat((y1_pred,y2_pred),dim=1)     # (size_batch, 2, num_x)

            y_pred_list.append(y_pred)
            if is_first:
                y_pred_sum = y_pred
            else:
                y_pred_sum = y_pred_sum + y_pred
            is_first = False
        y_pred_mean = y_pred_sum / self.num_models
        return y_pred_list, y_pred_mean

    def loss(self,y_pred_list, y):
        is_first = True
        for iy, y_pred in enumerate(y_pred_list):

            loss_R = torch.mean((y_pred-y)**2/torch.abs(y+0.02))
            if is_first:
                loss_R_sum = loss_R
            else:
                loss_R_sum = loss_R_sum + loss_R
            is_first = False
        
        loss_R_mean = loss_R_sum / self.num_models
        loss_G = torch.mean(torch.sum(self.latent**2,dim=1))  
        loss = loss_R_mean + 0.0 * loss_G
        return loss, loss_R, 0.0*loss