import pdb

import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
rs = npr.RandomState(0)


def shapes_and_num(layer_sizes):
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m + 1) * n for m, n in shapes)
    return shapes, N_weights


def unpack_layers(weights, layer_sizes):
    """ unpacks weights [ns, nw] into each layers relevant tensor shape"""
    shapes, _ = shapes_and_num(layer_sizes)
    n_samples = len(weights)
    for m, n in shapes:
        yield weights[:, :m * n].reshape((n_samples, m, n)), \
              weights[:, m * n:m * n + n].reshape((n_samples, 1, n))
        weights = weights[:, (m + 1) * n:]


def reshape_weights(weights, layer_sizes):
    layers = list(unpack_layers(weights, layer_sizes)) ## Added layers variable
    
    return layers


def bnn_predict(weights, inputs, layer_sizes, act):
    if len(inputs.shape)<3: inputs = np.expand_dims(inputs, 0)  # [1,N,D]
    weights = reshape_weights(weights, layer_sizes)
    for W, b in weights:
        #print(W.shape, inputs.shape)
        
        pdb.set_trace()

        outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
        inputs = act(outputs)
    return outputs


def gaussian_entropy(log_std):
    return 0.5 * log_std.shape[0] * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)  # [ns]

def sample_weights(params, N_samples):
    mean, log_std = params
    return rs.randn(N_samples, mean.shape[0]) * np.exp(log_std) + mean  # [ns, nw]

def sample_bnn(params, x, N_samples, layer_sizes, act, noise=0.0):
    bnn_weights = sample_weights(params, N_samples)
    f_bnn = bnn_predict(bnn_weights, x, layer_sizes, act)[:, :, 0]
    return f_bnn + noise * rs.randn(N_samples, x.shape[0])  # [ns, nd]


def vlb_objective(params, x, y, layer_sizes, n_samples, model_sd=0.1, act=np.tanh):
    """ estimates elbo = -H[q(w))]- E_q(w)[log p(D,w)] """
    mean, log_std = params
    weights = sample_weights(params, n_samples)
    entropy = gaussian_entropy(log_std)

    f_bnn = sample_bnn(params, x, n_samples,layer_sizes, act)
    log_likelihood = diag_gaussian_log_density(y.T, f_bnn, .1)
    log_prior = diag_gaussian_log_density(weights, 0, 1)

    return -entropy - np.mean(log_likelihood+log_prior)


def init_var_params(layer_sizes, scale=-5, scale_mean=1):
    _, num_weights = shapes_and_num(layer_sizes)
    return rs.randn(num_weights)*scale_mean, np.ones(num_weights)*scale  # mean, log_std

def build_toy_dataset(n_data=80, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets


def sample_data(n_data=20, noise_std=0.1, context_size=3):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-1.2,1.2,n_data)
    targets = inputs**3 + rs.randn(n_data) * noise_std
    return inputs[:, None], targets[:, None]

def train_bnn(inputs, targets, arch = [1, 20, 20, 1], lr=0.01, iters=100, n_samples=10, act=np.tanh):

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)


    def objective(params,t):
        return vlb_objective(params, inputs, targets, arch, n_samples, act)

    def callback(params, t, g):
        # Sample functions from posterior f ~ p(f|phi) or p(f|varphi)
        N_samples, nd = 5, 400
        plot_inputs = np.linspace(-8, 8, num=400)
        f_bnn = sample_bnn(params, plot_inputs[:,None], N_samples, arch, act)

        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'k.')
        ax.plot(plot_inputs, f_bnn.T, color='r')
        ax.set_ylim([-5, 5])
        plt.draw()
        plt.pause(1.0 / 60.0)

        print("ITER {} | OBJ {}".format(t, -objective(params, t)))

    var_params = adam(grad(objective), init_var_params(arch),
                      step_size=lr, num_iters=iters, callback=callback)

    return var_params



if __name__ == '__main__':
    np.random.seed(0)


    # Set up
    arch = [1, 10, 1]
    inputs, targets = sample_data()

    train_bnn(inputs, targets)


