import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import os
import math

version_higher = ( torch.__version__ >= "1.5.0" )

class Cal_eps_Adam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), num_batches = 391, weight_decay=0, amsgrad=False, 
                 weight_decouple = False, fixed_decay=False, rectify = False ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            
        defaults = dict(lr=lr, betas=betas, num_batches=num_batches, weight_decay=weight_decay, amsgrad=amsgrad,  
                        eps_upper=0.0, eps_lower=3.4028235e+38)
        super(Cal_eps_Adam, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in Adam')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in Adam')
        if amsgrad:
            print('AMS enabled in Adam')
    def __setstate__(self, state):
        super(Cal_eps_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            num_params = len(group['params'])
                    
            for kk, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
               
                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0

                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)


                # get current state variable
                exp_avg_var = state['exp_avg_var']

                state['step'] += 1

                bias_correction2 = 1 - beta2 ** state['step']

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])
                    
                # Update first and second moment running average
                grad_residual = grad**2
                exp_avg_var.mul_(beta2).add_(grad_residual, alpha = 1 - beta2)
                
                
                
                if (state['step']==group['num_batches']) : 
                    b = (exp_avg_var/bias_correction2).sqrt().flatten()
                    b=b[b!=0]
                    aux, _ = torch.sort(b)
                    
                    porcentage = 0.98
      
                    n_lower = int((1-porcentage)*aux.size()[0])
                    n_upper = int(porcentage*aux.size()[0])

                
                    group['eps_upper'] = max(group['eps_upper'], aux[n_upper]) 
                    group['eps_lower'] = min(group['eps_lower'], aux[n_lower]) 
                   
                    eps_upper =  10**round(math.log10( group['eps_upper'] ))
                    eps_lower =  10**round(math.log10( group['eps_lower'] ))

                    if (kk == num_params-1):
                        print('\n')
                        print('eps_upper: '+str(eps_upper))
                        print('eps_lower: '+str(eps_lower))    
                        print('\n')    


        return loss

