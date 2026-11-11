'''
Quick test, must be modified later.
'''

import torch

def x_exact(t, params = (5,500,1)):
    mu, k, m = params
    delta = mu/(2*m)
    omega0 = torch.sqrt(k)
    omega = torch.sqrt(omega0**2-delta**2)
    phi = 1/torch.tan(-delta/omega)
    A = 1/(2*torch.cos(phi))
    
    return torch.exp(-delta*t)*(2*A*torch.cos(phi+omega*t))

def phys_loss(xs, net, params):
    '''Assumes that the network has functions d2dx and ddx implemented'''
    
    mu, k, m = params
    term1 = net.d2dx(xs[:,None])
    term2 = mu/m*net.ddx(xs[:,None])
    term3 = k/m*net(xs)
    
    return torch.mean((term1+term2+term3)**2)

def IC_loss(net):
    '''Assumes that the network has property IC implemented'''
    IC_exact = torch.Tensor([1,0])[:,None]
    res = net.IC -IC_exact
    return torch.mean(res**2)

def collocation_generator(N, a=0,b=1):
    '''Collocation points from uniform distribution. on domain [a,b]'''
    return torch.distributions.uniform.Uniform(a,b).sample([N,1])