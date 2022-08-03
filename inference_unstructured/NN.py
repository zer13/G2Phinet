import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class G2Phi(nn.Module):
    def __init__(
            self, xz, xt, num_latent, num_types, alphas):
        super(G2Phi, self).__init__()

        self.xz = xz    # 1d np array
        self.xt = xt
        self.num_latent = num_latent
        self.num_types = num_types
        self.num_points_z = xz.shape[0]
        self.num_points_t = xt.shape[0]
        self.alphas = alphas

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sftpls = nn.Softplus()

        self.act = self.relu

        self.p = 128

        # branch encoder
        # 31x31x1
        self.conv1 = nn.Conv2d(2, 64, 5, stride=2)
        # 14x14x32
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2)
        # 6x6x64
        self.conv3 = nn.Conv2d(64, 64, 3)
        # 4x4x64
        self.conv4 = nn.Conv2d(64, 64, 3)
        # 2x2x64
        self.flat = nn.Flatten()
        # 256
        self.efc1 = nn.Linear(256,64)
        self.efc2 = nn.Linear(64,16)
        self.efc3 = nn.Linear(16,num_latent)

        self.encoder = nn.Sequential(
          self.conv1,
          self.act,
          self.conv2,
          self.act,
          self.conv3,
          self.act,
          self.conv4,
          self.act,
          self.flat,
          self.efc1,
          self.act,
          self.efc2,
          self.act,
          self.efc3
        )

        # encoder skip connection
        self.encoder_skip = nn.Sequential(nn.Flatten(),nn.Linear(2*self.num_points_z*self.num_points_t, num_latent))

        # branch decoder
        self.dfc1 = nn.Linear(num_latent+num_types, 48)
        self.dfc2 = nn.Linear(48, 48)
        self.dfc3 = nn.Linear(48, 48)
        self.dfc4 = nn.Linear(48, self.p*2)
        self.decoder = nn.Sequential(
            self.dfc1,
            self.act,
            self.dfc2,
            self.act,
            self.dfc3,
            self.act,
            self.dfc4
        )

        # decoder skip connection
        self.decoder_skip = nn.Sequential(nn.Linear(num_latent+num_types, self.p*2))

        # trunk
        self.tfc1 = nn.Linear(2, 48)
        self.tfc2 = nn.Linear(48, 48)
        self.tfc3 = nn.Linear(48, 48)
        self.tfc4 = nn.Linear(48, self.p*2)

        self.trunk = nn.Sequential(
            self.tfc1,
            self.tanh,
            self.tfc2,
            self.tanh,
            self.tfc3,
            self.tanh,
            self.tfc4,
            self.tanh
        )

        # trunk input
        Xt, Xz = np.meshgrid(self.xt, self.xz)
        X = np.concatenate((Xt.reshape(-1,1),Xz.reshape(-1,1)),axis=1)
        self.X = torch.tensor(X, dtype=torch.float, requires_grad=True)      # size_batch * 2; columns: Xz, Xt

        # b.*t bias
        bias = np.random.uniform(low=-1.0/np.sqrt(2), high=1.0/np.sqrt(2), size=(2,))
        self.bias = nn.Parameter(torch.tensor(bias, requires_grad=True))

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)

        # LR scheduler
        decay_rate = 0.5**(1/5000)          # after 4K epochs (see main), halve the lr every 10K epochs
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate)

    def forward_encoder(self, y):
        return self.encoder(y) + self.encoder_skip(y)
    
    def forward_decoder(self, latent, label):
        inp = torch.cat((label, latent),dim=1)
        return self.decoder(inp) + self.decoder_skip(inp)

    def forward_branch(self, y, label):
        latent = self.forward_encoder(y)
        outp = self.forward_decoder(latent, label)
        return outp, latent

    def forward_trunk(self):
        return self.trunk(self.X)

    def forward(self, y, label):
        # y: 4D
        #y = y[:,None,:,:]
        b, latent = self.forward_branch(y, label)      # (size_batch, 2*p)
        size_batch = b.shape[0]
        b1 = b[:,:self.p]
        b2 = b[:,self.p:]
        t = self.forward_trunk()               # (num_x, 2*p)
        num_x = t.shape[0]
        t1 = t[:,:self.p]
        t2 = t[:,self.p:]
        temp1 = (torch.matmul(b1,t1.T)+self.bias[0])*(self.X[:,0:1]**2+self.X[:,1:2]**2).T    # (size_batch, num_x)
        temp2 = (torch.matmul(b2,t2.T)+self.bias[1])*(self.X[:,0:1]**2+self.X[:,1:2]**2).T    # (size_batch, num_x)
        y1_pred = temp1.reshape([size_batch,1,self.num_points_z,self.num_points_t])
        y2_pred = temp2.reshape([size_batch,1,self.num_points_z,self.num_points_t])
        y_pred = torch.cat((y1_pred,y2_pred),dim=1)
        
        return y_pred, latent
    
    def losses(self, y, y_pred, latent, w):

        _, _, alpha_G = self.alphas     # alpha_R and alpha_M have been utilized in w

        # loss_R (Reconstruction)
        loss_R = torch.mean(w[:,:,None,None]*(y-y_pred)**2/torch.abs(y+0.02))

        # loss_G (Unsupervised Close to Origin)
        mean = torch.mean(latent,dim=0)
        var = torch.matmul(latent.T,latent)/latent.shape[0]

        KLD = 0.5*(torch.sum(mean**2)+torch.sum(torch.diagonal(var,0))-self.num_latent-torch.log(torch.det(var)))
        loss_G = KLD

        # loss total
        loss = loss_R + alpha_G*loss_G
        return loss, loss_R, loss_G