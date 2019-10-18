#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Henning Lange (helange@uw.edu)
"""


import torch

from torch import nn
from torch import optim

import numpy as np


class koopman(nn.Module):
    '''
    
    This implementation can utilize multiple GPUs when computing the global
    error surface but it can also be run just on a single CPU.
    '''
    
    
    def __init__(self, model_obj, sample_num = 12):
        '''
        Input:
            model_obj: an object that specifies the function f and how to optimize
                       it. The object needs to implement numerous function. See
                       models.py for some examples.
                       
            sample_num: number of samples from temporally local loss used to 
                        reconstruct the global error surface
        
        '''
        
        super(koopman, self).__init__()
        
        self.num_freq = model_obj.num_freq
        
        #Inital guesses for frequencies
        if self.num_freq == 1:
            self.omegas = torch.tensor([0.2])
        else:
            self.omegas = torch.linspace(0.01,0.5,self.num_freq)
            
        self.model_obj = nn.DataParallel(model_obj)
        
        #number of samples to reconstruct the global error surface
        self.sample_num = sample_num
        
        #if you run out of memory, decrease batch_per_gpu
        self.batch_per_gpu = 4
        self.num_gpu = 3
        
        
        
    def unroll_i(self, xt, i):
        
        '''
        Given a dataset, this function samples temporally local loss functions
        w.r.t. the i-th entry of omega
        
        Input:
            xt: dataset whose first dimension is time 
                dimensions: [T, ...]
                type: numpy.array
                
            i: index of the entry of omega
                type: int
            
        Output:
            errs: matrix that contains temporally local losses between [0,2pi/t]
                  dimensions: [T, sample_num]
        
        '''
        
        if type(xt) == np.ndarray:
            xt = torch.from_numpy(xt)
            
        t = torch.arange(xt.shape[0])+1
        
        errors = []
        
        batch = self.batch_per_gpu * self.num_gpu
        
        for j in range(t.shape[0]//batch):
            
            torch.cuda.empty_cache()

            ts = t[j*batch:(j+1)*batch] 
            
            o = torch.unsqueeze(self.omegas, 0)
            ts = torch.unsqueeze(ts,-1).type(torch.get_default_dtype())
            
            ts2 = torch.arange(self.sample_num, dtype=torch.get_default_dtype())
            ts2 = ts2*2*np.pi/self.sample_num

            ts2 = ts2*ts/ts #essentially reshape
            
            ys = []
            
            for iw in range(self.sample_num):
                wt = ts*o
                
                wt[:,i] = ts2[:,iw]
                
                y = torch.cat([torch.cos(wt), torch.sin(wt)], dim=1)
                ys.append(y)
            
            ys = torch.stack(ys, dim=-2).data
            x = torch.unsqueeze(xt[j*batch:(j+1)*batch],dim=1)
            
            
            loss = self.model_obj(ys, x)            
            errors.append(loss.cpu().detach().numpy())
            
        torch.cuda.empty_cache()
        
        return np.concatenate(errors, axis=0)
    
    
    def optimize_omega(self, xt, i, verbose=False):
        
        '''
        Given a dataset, this function updates the i-th entry of omega
        
        Input:
            xt: dataset whose first dimension is time 
                dimensions: [T, ...]
                type: numpy.array
                
            i: index of the entry of omega
                type: int
            
        Output: (mostly for debugging)
            E: the global loss surface in time domain
            E_ft: the global loss surface in freq domain
            errs: temporally local loss function
        
        '''
        
        errs = self.unroll_i(xt,i)
        ft_errs = np.fft.fft(errs)
        
        E_ft = np.zeros(xt.shape[0]*self.sample_num).astype(np.complex64)
        
        for t in range(1,ft_errs.shape[0]+1):
            E_ft[np.arange(self.sample_num)*t] += ft_errs[t-1,:self.sample_num]
            
        E_ft = np.concatenate([E_ft, np.conj(np.flip(E_ft))])[:-1]
            
        E = np.fft.ifft(E_ft)
        omegas = np.linspace(0,1,len(E))
        
        idxs = np.argsort(E[:len(E_ft)//2])
        
        omegas_actual = self.omegas.cpu().detach().numpy()
        omegas_actual[i] = -1
        found = False
        
        j=0
        while not found:
            
            if idxs[j]>5 and np.all(np.abs(2*np.pi/omegas_actual - 1/omegas[idxs[j]])>1):
                found = True
                if verbose:
                    print('Setting ',i,'to',1/omegas[idxs[j]])
                self.omegas[i] = torch.from_numpy(np.array([omegas[idxs[j]]]))
                self.omegas[i] *= 2*np.pi
            
            j+=1
            
        return E, E_ft, errs
    
    
    
    
    def optimize_refine(self, xt, batch_size=128, verbose=False):
        
        '''
        Given a dataset, this function improves parameters of f and performs 
        SGD improvements on omega.
        
        Input:
            xt: dataset whose first dimension is time 
                dimensions: [T, ...]
                type: numpy.array
                
            batch_size: batch size for SGD
                type: int
            
        Output:
            loss
        
        '''
        
        T = xt.shape[0]
        
        omega = nn.Parameter(self.omegas)
        
        opt = optim.SGD(self.model_obj.parameters(), lr=1e-3)
        opt_omega = optim.SGD([omega], lr=1e-5/T)
        
        
        T = xt.shape[0]
        t = torch.arange(T)
        
        losses = []
        
        for i in range(len(t)//batch_size):
            
            ts = t[i*batch_size:(i+1)*batch_size]
            o = torch.unsqueeze(omega, 0)
            ts_ = torch.unsqueeze(ts,-1).type(torch.get_default_dtype()) + 1
            
            xt_t = torch.from_numpy(xt[ts.cpu().numpy(),:]).cuda()
            
            wt = ts_*o
            
            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)            
            loss = torch.mean(self.model_obj(k, xt_t))
            
            opt.zero_grad()
            opt_omega.zero_grad()
            
            loss.backward()
            
            opt.step()
            opt_omega.step()
            
            losses.append(loss.cpu().detach().numpy())
            
        if verbose:
            print('Setting to', 2*np.pi/omega)
            
        self.omegas = omega.data
                

        return np.mean(losses)
    
    
    
    def fit(self, xt, iterations = 10, interval = 5, verbose=False):
        '''
        Given a dataset, this function alternatingly optimizes omega and 
        parameters of f.
        
        Input:
            xt: dataset whose first dimension is time 
                dimensions: [T, ...]
                type: numpy.array
                
            iterations: number of iterations over the dataset
                type: int
            
            interval: the interval at which omegas are updated, i.e. if 
                      interval is 5, then omegas are updated every 5 iterations
                type: int
            
        '''
        
        assert(len(xt.shape))
    
        for i in range(iterations):
            
            if i%interval == 0:
                for k in range(self.num_freq):
                    self.optimize_omega(xt, k, verbose=verbose)
            
            if verbose:
                print('Iteration ',i)
                print(2*np.pi/self.omegas)
            
            l = self.optimize_refine(xt, verbose=verbose)
            if verbose:
                print('Loss: ',l)
            
            
            
    def predict(self, T):
        
        '''
        After learning, this function will provide predictions
        
        Input:
            T: prediction horizon
                type: int
                
        Output:
            x_pred: predictions
                dimensions: [T,...]
                type: numpy.array
        
        '''
        
        t = torch.arange(T)+1
        ts_ = torch.unsqueeze(t,-1).type(torch.get_default_dtype())

        o = torch.unsqueeze(self.omegas, 0)
        k = torch.cat([torch.cos(ts_*o), torch.sin(ts_*o)], -1)
        mu = self.model_obj.module.decode(k.cuda())

        return mu.cpu().detach().numpy()





class model_object(nn.Module):
    
    def __init__(self, num_freq):
        super(model_object, self).__init__()
        self.num_freq = num_freq
        
    
    
    def forward(self, y, x):
        '''
        Forward computes the error.
        
        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
                
            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        '''
        
        
        raise NotImplementedError()
    
    def decode(self, y):
        '''
        Evaluates f at temporal snapshots y
        
        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
                
            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        '''
        raise NotImplementedError()




class fully_connected_mse(model_object):
    
    
    def __init__(self, x_dim, num_freqs, n):
        super(fully_connected_mse, self).__init__(num_freqs)
        
        self.l1 = nn.Linear(2*num_freqs, n)
        self.l2 = nn.Linear(n,32)
        self.l3 = nn.Linear(32,x_dim)
        
        
    def decode(self, x):
        o1 = nn.Tanh()(self.l1(x))
        o2 = nn.Tanh()(self.l2(o1))
        o3 = self.l3(o2)
        
        return o3
        
        
    def forward(self, y, x):
        
        xhat = self.decode(y)
        return torch.mean((xhat-x)**2, dim=-1)
    
        