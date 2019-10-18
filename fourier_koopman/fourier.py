#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Henning Lange (helange@uw.edu)
"""

import numpy as np
import torch


class fourier:
    
    def __init__(self, num_freqs, use_cpu = True):
        '''
        Input:
            num_freqs: number of frequencies assumed to be present in data
                type: int
        
        '''
        
        self.num_freqs = num_freqs
        self.use_cpu = use_cpu
    
    
    def initial_guess(self, xt):
        '''
        Given a dataset, this function performs the initial guess of the 
        frequencies contained in the dataset
        
        Input:
            xt: dataset whose first dimension is time 
                dimensions: [T, ...]
                type: numpy.array
        '''
        
        k = self.num_freqs
        self.freqs = []
        
        for i in range(k):
        
            N = len(xt)
            
            if len(self.freqs) == 0:
                residual = xt
            else:
                t = np.expand_dims(np.arange(N)+1,-1)
                freqs = np.array(self.freqs)
                Omega = np.concatenate([np.cos(t*2*np.pi*freqs),
                                        np.sin(t*2*np.pi*freqs)],-1)
                self.A = np.dot(np.linalg.pinv(Omega), xt)
                
                pred = np.dot(Omega,self.A)
                
                residual = pred-xt
            
            
            ffts = 0
            for j in range(xt.shape[1]):
                ffts += np.abs(np.fft.fft(residual[:,j])[:N//2])
        
            
            w = np.fft.fftfreq(N,1)[:N//2]
            idxs = np.argmax(ffts)
            
            self.freqs.append(w[idxs])
            
            
            t = np.expand_dims(np.arange(N)+1,-1)
            
            Omega = np.concatenate([np.cos(t*2*np.pi*self.freqs),
                                    np.sin(t*2*np.pi*self.freqs)],-1)
    
            self.A = np.dot(np.linalg.pinv(Omega), xt)

    
    
    def refine(self, data, iterations = 1000, learning_rate = 3E-9, verbose=False):
        
        '''
        Given a dataset, this function improves the initial guess by SGD
        
        Input:
            xt: dataset whose first dimension is time 
                dimensions: [T, ...]
                type: numpy.array
                
            iterations: the number of iterations over the dataset
                tyoe: int
                
            learning_rate: the step size used for SGD.
                NOTE: Because gradients grow with time, learning rate should
                be a function of T
        '''
        
        
        if False:
            A = torch.tensor(self.A, requires_grad=False).cuda()
            freqs = torch.tensor(self.freqs, requires_grad=True).cuda()
            data = torch.from_numpy(data).cuda()
        else:
            A = torch.tensor(self.A, requires_grad=False).cpu()
            freqs = torch.tensor(self.freqs, requires_grad=True).cpu()
            data = torch.from_numpy(data).cpu()

        o2 = torch.optim.SGD([freqs], lr=learning_rate)
        
        t = torch.unsqueeze(torch.arange(len(data))+1,-1).type(torch.get_default_dtype())
        
        loss = 0
        
        for i in range(iterations):
            
            Omega = torch.cat([torch.cos(t*2*np.pi*freqs),
                               torch.sin(t*2*np.pi*freqs)],-1)
    
            A = torch.matmul(torch.pinverse(Omega.data), data)
    
            xhat = torch.matmul(Omega,A)
            loss = torch.mean((xhat-data)**2)
            
            o2.zero_grad()
            loss.backward()
            o2.step()
            
            loss = loss.cpu().detach().numpy()
            if verbose:
                print(loss)
            
            
            
        self.A = A.cpu().detach().numpy()
        self.freqs = freqs.cpu().detach().numpy()
        
        
        
    def fit(self, data, learning_rate = 1E-5, iterations = 1000, verbose=False):
        
        self.initial_guess(data)
        self.refine(data, iterations = iterations, 
                    learning_rate = learning_rate/data.shape[0],
                    verbose = verbose)
        
        return self.freqs
    
    
    
    def predict(self, T):
        
        t = np.expand_dims(np.arange(T)+1,-1)
        Omega = np.concatenate([np.cos(t*2*np.pi*self.freqs),
                                np.sin(t*2*np.pi*self.freqs)],-1)
        
        return np.dot(Omega,self.A)
