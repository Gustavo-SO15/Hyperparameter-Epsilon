import math
import torch
from torch.optim.optimizer import Optimizer

version_higher = ( torch.__version__ >= "1.5.0" )

class TAdam(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam

    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, eps2=1,
                 weight_decay=0, amsgrad=False, weight_decouple = False, fixed_decay=False, rectify = False ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,eps2=eps2,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(TAdam, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMS enabled in AdaBelief')
    def __setstate__(self, state):
        super(TAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                state['epsd'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data,
                                   memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data,
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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
               
                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    state['epsd'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data,
                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                #beta2 = 0.9
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                #bias_correction3 = 1 - 0.99999 ** state['step']
                #beta2 = 1 - 0.999/state['step']  
                
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
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = (grad)**2
                exp_avg_var.mul_(beta2).add_(grad_residual, alpha = 1 - beta2)

               # exp_avg_var.mul_(beta2).add_(0.5*grad_residual+0.5*grad_residual.mean(), alpha = 1 - beta2)
                
               # if state['step'] <=  200:
               #     print('mean:'+str((grad**2).mean()))
               #     print('min:'+str((grad**2).min()))
               #     print('max:'+str((grad**2).max()))
            
               # if 39200 <= state['step'] <=  39220:
               #     print('mean:'+str((grad**2).mean()))
               #     print('min:'+str((grad**2).min()))
               #     print('max:'+str((grad**2).max()))

               
                
                
                #if 664<=state['step']<= 4*663:  
                  #  aux_d = grad_residual.mean() # torch.sqrt((grad**2).mean()*(grad**2).min())
                    #if aux_d == 0:
                    #    aux_d = (exp_avg_var).mean()
                    #aux_d = 10**math.floor(math.log10(aux_d))
                   # state['epsd'] = ((state['step']-1)*state['epsd'] + aux_d)/state['step']
                    #group['eps'] = min(group['eps'], aux_d)
                    #if state['epsd'] >= 0.01:
                    #    print('*************************')
                #    aux = 1e-5 *bias_correction2     #group['eps']*bias_correction2*1000  
                    #print('var: '+ str(torch.var(grad)))
                  #  print('mean: '+ str(torch.mean(grad**2)))
                   # varx = (grad**2).mean()
                   # group['eps'] = min(group['eps'],varx)
                  #  aux = 1
                  #  print('eps: '+str(aux))
                  #  print('eps: '+str(group['eps']))
                  #  print('var: '+str(varx))
                  #  print('mean: '+str(torch.mean(grad)))
                  #  print('mean1: '+str(torch.mean(grad.abs())))
                  #  print('std1: '+str(torch.std(grad.abs())))
                  #  print('var1: '+str(torch.var(grad.abs())))
                  #  mea=10**round(math.log10(torch.mean(grad_residual)))
                    
                  #  group['eps'] = min( group['eps'], mea)
                    
                  #  print((grad_residual>=1e-6).sum()/grad_residual.nelement())
                  #  print('step: '+str(state['step']))
                  ##  print((grad_residual>=group['eps']).sum()/grad_residual.nelement())
                  ##  print((grad_residual>=mea).sum()/grad_residual.nelement())
                 ##   print('mean1: '+str(mea))
                 ##   print(grad_residual.nelement())
                    
                   # group['eps'] = group['eps']+round(math.log10(mea))
                   # if state['step']%391==0:
                   #     print('eps-est: '+str(group['eps']))
                   # aux = 1
                    #print('mean2: '+str(torch.mean(exp_avg_var)))
                  #  print('std: '+str(torch.std(grad**2)))
                   # print('var: '+str(torch.var(grad**2)))
                  #  print('min: '+str((grad**2).min()))
                  #  print('max: '+str((grad**2).max()))
                   # print(exp_avg_var.mean())
               # else:
                #    aux = 1

                #if state['step']== 391:  
                   # print('mean: '+str((grad**2).mean()))
                    #print('min: '+str((grad**2).min()))
                   # print('max: '+str((grad**2).max()))
                 #   print('eps: '+str(state['epsd']))
                    
               # aux_d = bias_correction2*grad_residual.mean() # torch.sqrt((grad**2).mean()*(grad**2).min())       
              #  aux_d = bias_correction2*grad_residual.median() # torch.sqrt((grad**2).mean()*(grad**2).min())
                if state['step']<= 1: 
                    aux3, _ = torch.sort(grad_residual.flatten())
                    porcentage = 0.95
                    
                    n = int((1-porcentage)*aux3.size()[0])
                    aux_d = bias_correction2*aux3[n]#grad**2).mean()
                    
                        #if aux_d == 0:
                        #    aux_d = (exp_avg_var).mean()
                        #aux_d = 10**math.floor(math.log10(aux_d))
                    state['epsd'] = ((state['step']-1)*state['epsd'] + aux_d)/state['step']
                       # state['epsd'] = aux_d

                    aux1 = max(state['epsd'], group['eps'])#*math.sqrt(state['step'])
                 #   aux1 = 10**math.floor(math.log10(aux1))
                    group['eps']= aux1#*math.sqrt(state['step'])

                    aux2 = min(state['epsd'], group['eps2'])#*math.sqrt(state['step'])
                #    aux2 = 10**math.floor(math.log10(aux2))
                    group['eps2']= aux2

                aux = math.sqrt(group['eps']*group['eps2'])
                aux = 10**math.floor(math.log10(aux))

                if state['step']<= 1: 
                    print('max :'+str(group['eps']))
                    print('min :'+str(group['eps2']))
                    print(aux)
                   # print(state['epsd'] )
                   # print(aux3.size()[0])
                   # print(aux2)
                #    print(aux)
                
                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.add_(aux).sqrt() / math.sqrt(bias_correction2))
                else:
                    denom = (exp_avg_var.add_(aux).sqrt() / math.sqrt(bias_correction2))
                    #print(exp_avg_var.mean())

                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value = -step_size)

                else:# Rectified update
                    # calculate rho_t
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (
                            1.0 - beta2 ** state['step'])

                    if state['rho_t'] > 4: # perform Adam style update if variance is small
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t
                        rt = math.sqrt(rt)

                        step_size = rt * group['lr'] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else: # perform SGD style update
                        p.data.add_(exp_avg, alpha =  -group['lr'])

        return loss

