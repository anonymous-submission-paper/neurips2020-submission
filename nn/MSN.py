import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import time
from copy import deepcopy


class MSN(nn.Module):
    """ 
        Mutual Information Statistic Network
    """
    def __init__(self, architecture, dim_y):
        super().__init__()
        self.score_layer = ScoreLayer(architecture[-1], dim_y, 100)
        self.encode_layer = EncodeLayer(architecture)
        self.estimator = 'JSD'          # <-- ['DV','f-div','JSD']
        
    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return self.encode_layer(x)
        
    def MI(self, z, y, n=10):
        # [A]. Donsker-Varadhan Representation (MINE, ICML'18)
        if self.estimator == 'DV':
            z_sample=z
            y_sample=y
            pred_zy = self.score_layer(z_sample, y_sample)
            ZY = torch.zeros(n, 1).to(z.device)        
            for i in range(n):
                idx = torch.randperm(len(y))
                y_shuffle=y_sample[idx]
                pred_z_y = self.score_layer(z_sample, y_shuffle)
                ZY[i] = torch.exp(pred_z_y).mean()
            mi = pred_zy.mean() - ZY.mean().log()
        # [B]. f-divergence (f-GAN, NIPS'17)
        if self.estimator == 'f-div':
            z_sample=z
            y_sample=y
            pred_zy = self.score_layer(z_sample, y_sample)
            ZY = torch.zeros(n, 1).to(z.device)        
            for i in range(n):
                idx = torch.randperm(len(y))
                y_shuffle=y_sample[idx]
                pred_z_y = self.score_layer(z_sample, y_shuffle)
                ZY[i] = torch.exp(pred_z_y-1).mean()
            mi = pred_zy.mean() - ZY.mean()
        # [C]. Jensen-shannon divergence (DeepInfoMax, ICLR'19)
        if self.estimator == 'JSD':
            z_sample=z
            y_sample=y
            pred_zy = self.score_layer(z_sample, y_sample)
            ZY = torch.zeros(n, 1).to(z.device)        
            for i in range(n):
                idx = torch.randperm(len(y))
                y_shuffle=y_sample[idx]
                pred_z_y = self.score_layer(z_sample, y_shuffle)
                ZY[i] = torch.exp(pred_z_y).mean()
            A, B = -F.softplus(-pred_zy), F.softplus(ZY)
            mi = A.mean() - B.mean()
        return mi

    def learn(self, x, y):     
        optimizer_S = torch.optim.Adam(self.score_layer.parameters(), lr=1e-4, weight_decay=0e-3)
        optimizer_E = torch.optim.Adam(self.encode_layer.parameters(), lr=1e-4, weight_decay=1e-3)
        bs = 5000 if len(x)>5000 else 850               # <-- use large batch size to reduce variance
        T = 4000

        # divide train & val
        n = len(x)
        n_val = int(0.85*n)
        idx = torch.randperm(n)
        x_train, y_train = x[idx[0:n_val]], y[idx[0:n_val]]
        x_val, y_val = x[idx[n_val:n]], y[idx[n_val:n]]
        
        # learn!
        n_batch = int(len(x_train)/bs)
        best_val_loss, best_model_state_dict, no_improvement = math.inf, None, 0
        for t in range(T):
            
            # shuffle 
            idx = torch.randperm(len(x_train))
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            
            # optimize
            for i in range(len(x_chunks)):
                optimizer_S.zero_grad()
                optimizer_E.zero_grad()
                loss = -self.MI(self.encode(x_chunks[i]), y_chunks[i], n=100)
                loss.backward()
                optimizer_S.step()
                optimizer_E.step()
                
            # early stopping if val loss does not improve after some epochs
            loss_val = -self.MI(self.encode(x_val), y_val, n=200)
            improved = loss_val.item() < best_val_loss
            no_improvement = 0 if improved else no_improvement + 1
            best_val_loss = loss_val.item() if improved else best_val_loss     
            best_model_state_dict = deepcopy(self.state_dict()) if improved else best_model_state_dict
            if no_improvement >= 100: break
                
            # report
            if t%int(T/20) == 0: print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item())
        
        # return the best model
        self.load_state_dict(best_model_state_dict)
        print('best val loss=', best_val_loss)
        return loss.item()
        
        
class ScoreLayer(nn.Module): 
    def __init__(self, dim_x, dim_y, n_hidden):
        super().__init__()        
        self.fc1 = nn.Linear(dim_x, n_hidden)
        self.fc2 = nn.Linear(dim_y, n_hidden)
        self.main = nn.Sequential(
            *(nn.Linear(n_hidden, n_hidden, bias=True) for i in range(1)),
            #nn.Dropout(p=0.2)
        )
        self.out = nn.Linear(n_hidden, 1)
        
    def forward(self, x, y):
        h = self.fc1(x) + self.fc2(y)
        for layer in self.main:
            h = F.relu(layer(h))
        out = self.out(h)
        return out
            
            
class EncodeLayer(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.main = nn.Sequential(
           *(nn.Linear(architecture[i], architecture[i+1], bias=True) for i in range(len(architecture)-2)),
           #nn.Dropout(p=0.2)
        )
        self.out = nn.Linear(architecture[-2], architecture[-1], bias=True)
        self.N_layers = len(architecture) - 1
            
    def forward(self, x):
        for layer in self.main: x = F.relu(layer(x))
        return self.out(x)