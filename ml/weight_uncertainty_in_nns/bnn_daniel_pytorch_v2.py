import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

# NN utility function

def get_layer_sizes_from_arch(arch):
   """get layer_sizes from architechture layer sizes description"""
   
   layer_sizes = [[m,n] for m,n in  zip(arch[:-1], arch[1:])]
   
   return layer_sizes 

def get_num_weights_from_arch(arch):

    layer_sizes = get_layer_sizes_from_arch(arch)

    num_weights = sum((m+1)*n for m,n in layer_sizes)

    return num_weights


# simple bnn to sample from, includes the fact that each want to have sample outputs & weights

class BNN():

    def __init__(self, arch, mean, log_std):
        self.layer_sizes = get_layer_sizes_from_arch(arch)
        
        self.params = (mean, log_std)

    def sample_weights(self, N_samples):
        mean, log_std = self.params
    
        epsilons = torch.randn(N_samples, mean.shape[0]) 
        log_std_expanded = log_std.expand(N_samples, *log_std.shape)
        mean_expanded = mean.expand(N_samples, *mean.shape)

        return epsilons*log_std_expanded.exp() + mean_expanded

    def unpack_layers(self, weights):
        n_samples = len(weights)

        for m,n in self.layer_sizes:
            yield weights[:, :m*n].reshape((n_samples, m, n)), \
              weights[:, m*n : m*n + n].reshape(n_samples, 1, n)

            weights = weights[:, (m+1) * n:]

    def reshape_weights(self, weights):
        layer_weights = list(self.unpack_layers(weights))

        return layer_weights

    def bnn_predict(self, weights, inputs, act):

        #inputs = inputs[None, :, :] # [1, N, D]
        weights = self.reshape_weights(weights)

        for W,b in weights:
            #outputs = torch.Tensor(np.einsum('mnd,mdo->mno', inputs, W)) + b
           
            if len(inputs.shape) == 2:
                num_weight_samples = W.shape[0]
                inputs = inputs.expand(num_weight_samples, *inputs.shape)

            #pdb.set_trace()
            
            outputs = inputs.bmm(W) + b
            inputs = act(outputs)

        return outputs 


    def sample_bnn(self, x, N_samples, act = F.tanh):
        bnn_weights = self.sample_weights(N_samples)

        f_bnn = self.bnn_predict(bnn_weights, x, act)[:, :, 0]

        return f_bnn, bnn_weights



# bnn_wrapper with mean & std parameters to optimize

class BNN_wrapper(nn.Module):

    def __init__(self, arch):
        super(BNN_wrapper, self).__init__()

        # initialize posterior parameters
        self.num_weight_samples = get_num_weights_from_arch(arch)
        self.mean = nn.Parameter(torch.randn(self.num_weight_samples))
        #self.log_std = nn.Parameter(torch.zeros(self.num_weight_samples))
        self.log_std = nn.Parameter(torch.ones(self.num_weight_samples) * -5)

    def forward(self, inputs, N_samples, act=F.relu):
        bnn_func = BNN(arch, self.mean, self.log_std)

        outputs, sampled_weights = bnn_func.sample_bnn(inputs, N_samples, act)

        return outputs, sampled_weights


# simple vlb_objective like in Daniel's file

def gaussian_entropy(log_std):
    return 0.5 * log_std.shape[0] * (1.0 + np.log(2*np.pi)) + log_std.sum()

def diag_gausian_log_density(x, mu, log_std):
    n = x.shape[0]

    #batch_size = mu.shape[0]
    #x_expanded = x.expand(batch_size, *x.shape)
    
    std = np.exp(log_std)

    #pdb.set_trace()

    return -0.5*n*(np.log(2*np.pi) + log_std) - ((x - mu)**2/(2* (std**2))).sum(dim=1)

def vlb_objective(params, y, mu_y, sample_weights):
    mean, log_std = params
    
    entropy = gaussian_entropy(log_std)

    log_likelihood = diag_gausian_log_density(y.t(), mu_y, 0.1)
    log_prior = diag_gausian_log_density(sample_weights, 0, 1)

    return - entropy - (log_likelihood + log_prior).mean()


# Train & Test Utilities

def plot_update(ax, params, inputs, targets, N_samples, arch):

    # get data to plot
    plot_inputs = torch.Tensor(np.linspace(-8,8, num=400)).view(-1,1)
    
    mean, log_std = params
    bnn_func = BNN(arch, mean, log_std)
   
    outputs, _ = bnn_func.sample_bnn(plot_inputs, N_samples)
   
    # Numpy versions of data to plot
    inputs_numpy = inputs.detach().numpy()
    output_means = outputs.mean(0).detach().numpy()
    plot_inputs_numpy = plot_inputs.detach().numpy()

    # plot data
    plt.cla()
    ax.plot(inputs_numpy, targets.numpy(), 'k.')
    ax.plot(plot_inputs_numpy, output_means, color='r')
    ax.set_ylim([-5,5])
    plt.draw()
    plt.pause(1.0/60.0)

# train & test functions

def train(model, n_data, batch_size, num_mc_weight_samples=10, n_iter=100):
    model.train()

    params = (model.mean, model.log_std)
    inputs, targets = sample_data(n_data)

    batch_size = 3

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)

    for i in range(0, n_data, batch_size):
        input_ = inputs[i:i+batch_size]
        target = targets[i:i+batch_size]
    
    #for ii in range(n_iter):
    #    input_ = inputs
    #    target = targets

        #optimizer.zero_grad()

        outputs, sample_weights = model.forward(input_, num_mc_weight_samples)
        train_loss = vlb_objective(params, target, outputs, sample_weights)

        print('parameters at iteration')
        print([p.sum().item() for p in model.parameters()])

        train_loss.backward()
        optimizer.step()

        print('train_loss', train_loss.item()) # why take -?
        print('mean', model.mean.mean().item())
        print('log_std', model.log_std.mean().item())

        plot_update(ax, params, inputs, targets, num_mc_weight_samples, arch)

def test(model, n_data=20, num_mc_weight_samples = 10):
    model.eval()

    with torch.no_grad():
        inputs, targets = sample_data(n_data)
        outputs, sampled_weights = model.forward(inputs, num_mc_weight_samples)

        test_loss = ((outputs.mean(0) - targets)**2).mean()

        print('test_loss', test_loss.item())



# data loader functions

def build_toy_dataset(n_data=80, noise_std=0.1):
    #rs = npr.RandomState(0)
    inputs	= np.concatenate([np.linspace(0, 3, num=n_data/2),	  
                      np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + torch.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs	= inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets

def sample_data(n_data=20, noise_std=0.1, context_size=3):
    #rs = npr.RandomState(0)

    inputs	= torch.Tensor(np.linspace(-1.2,1.2,n_data))
    targets = inputs**3 + torch.randn(n_data) * noise_std
    return inputs[:, None], targets[:, None]


# main functions

if __name__ == '__main__':
    torch.manual_seed(0)

    device = torch.device('cpu')

    arch = [1,20,20,1]

    # get the model instance
    model = BNN_wrapper(arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    n_data = 1000
    batch_size = 20
    train(model, n_data, batch_size)
    test(model)

    # Test
    
    #num_weights = get_num_weights_from_arch(arch)
    #mean = torch.zeros(num_weights)
    #log_std = torch.ones(num_weights)
    #bnn_func = BNN(arch, mean, log_std)

    inputs, targets = sample_data(20)
    N_samples = 2

    #bnn_func.sample_bnn(inputs, N_samples)

    #bnn = BNN_wrapper(arch)
    
    #params = (bnn.mean, bnn.log_std)
    #outputs, weights = bnn.forward(inputs, N_samples)

    #loss = vlb_objective(params, targets, outputs, weights)

    pdb.set_trace()
