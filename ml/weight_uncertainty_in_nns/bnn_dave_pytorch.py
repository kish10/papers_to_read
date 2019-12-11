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


# Model

class BNN(nn.Module):

	def __init__(self, arch, mean, log_std):
		super(BNN, self).__init__()

		# define linear components of the layers
		self.layer_sizes = get_layer_sizes_from_arch(arch)

		self.linears = nn.ModuleList(
			[nn.Linear(*layer_size) for layer_size in self.layer_sizes]
		)

		# store posterior parameters
		self.num_weight_samples = sum((m + 1)*n for m,n in self.layer_sizes)
		self.mean =  mean
		self.log_std = log_std
   

	def get_num_weights_from_layer_sizes(layer_sizes):
		# NOT used
		# NOTE the +1 accounts for the bias in the subsequent layer

		num_weights = sum((m+1)*n for m,n in layer_sizes)

		return num_weights


	def unpack_sampled_weights(self, sampled_weights):
		"""unpacks weights '[ns nw]' (?) into each layers relevant tensor shape """
		# NOTE Pytorch defines weights & bias separately
		# NOTE Pytorch transposes weight matrix relative to the shape used in the definition of the linear layers

		#layer_sizes = BNN.get_layer_sizes_from_arch(self.arch)
		num_weight_samples = len(sampled_weights)
		num_layer_weights = [m*n for m, n in self.layer_sizes]
		num_layer_biases = [n for _,n in self.layer_sizes]

		all_layer_weights = []
		for ii in range(len(num_layer_weights)):
			start_index = sum(num_layer_weights[:ii])
			stop_index = sum(num_layer_weights[:ii+1])

			layer_weights =  nn.Parameter(
				sampled_weights[start_index:stop_index]
				.view(*self.layer_sizes[ii][::-1])
			)

			all_layer_weights += [layer_weights]

		bias_start_index = stop_index

		all_layer_biases = []
		for jj in range(len(num_layer_biases)):
			start_index = bias_start_index + sum(num_layer_biases[:jj])
			stop_index = bias_start_index + sum(num_layer_biases[:jj + 1])

			layer_biases = nn.Parameter(
				sampled_weights[start_index:stop_index]
				.view(-1)
			)

			all_layer_biases += [layer_biases]

		return all_layer_weights, all_layer_biases

	def sample_weights(self):
		epsilons = torch.randn(self.num_weight_samples, 1)
		sampled_weights = self.mean + self.log_std.exp()*epsilons

		return sampled_weights

	
	def bnn_predict(self, layer_weights, layer_biases, inputs, act):
	   
		for lin, lin_w, lin_b in zip(self.linears, layer_weights, layer_biases):
			lin.weight = lin_w
			lin.bias = lin_b

			outputs = lin(inputs)
			inputs = act(outputs)
		return outputs

	def forward(self, inputs, act=F.tanh):

		# reshape images into tensor with 784 collumns
		#inputs = inputs.view(-1, 784)

		# get sampled weights
		sampled_weights = self.sample_weights()
		unpacked_weights = self.unpack_sampled_weights(sampled_weights)
		sampled_weights_unpacked, sampled_biases_unpacked = unpacked_weights

		# set weights & generate an output
		outputs = self.bnn_predict(
			sampled_weights_unpacked,
			sampled_biases_unpacked,
			inputs, act)

		return outputs, sampled_weights


	

class BNN_simple():

	def __init__(self, arch, mean, log_std):
		self.layer_sizes = get_layer_sizes_from_arch(arch)

		self.params = (mean, log_std)
	
	def sample_weights(self, N_samples):
		mean, log_std = self.params

		return torch.randn(N_samples, mean.shape[0]) * log_std.exp() + mean

	def unpack_layers(weights):
		n_samples = len(weights)

		for m,n in self.layer_sizes:
			yield weights[:, :m*n].reshape((n_samples, m, n)), \
				  weights[:, m*n : m*n + n].reshape(n_samples, 1, n)

			weights = weights[:, (m+1) * n]

	def reshape_weights(self, weights):
		layer_weights = list(self.unpack_layers(weights))

		return layer_weights

	def bnn_predict(weights, inputs, act):
		weights = reshape_weights(weights)

		for W,b in weights:
			outputs = np.einsum('mnd,mdo->mno', inputs, w) + b
			inputs = act(outputs)

		return outputs 

	def sample_bnn(self, x, N_samples, layer_sizes, act):
		bnn_weights = self.sample_weights(N_samples)
		f_bnn = bnn_predict(bnn_weights, x, layer_sizes, act)[:, :, 0]

		return f_bnn, bnn_weights


class BNN_wrapper(nn.Module):
	def __init__(self, arch):
		super(BNN_wrapper, self).__init__()

		self.layer_sizes = get_layer_sizes_from_arch(arch)

		# initialize posterior parameters
		self.num_weight_samples = sum((m + 1)*n for m,n in self.layer_sizes)
		self.mean =  nn.Parameter(torch.randn(self.num_weight_samples, 1))
		self.log_std = nn.Parameter(torch.zeros(self.num_weight_samples, 1))

	def forward(self, inputs, N_samples, act=F.relu):

		with torch.no_grad():
			sample_bnn = BNN(arch, self.mean, self.log_std)
			outputs, sampled_weights = sample_bnn.forward(inputs, act)

		#with torch.no_grad():
		#	bnn = BNN_simple(arch, self.mean, self.log_std)
		#	outputs, sampled_weights = bnn.sample_bnn(inputs, N_samples)

		return outputs, sampled_weights


 # loss function

def gaussian_entropy(log_std):
	return 0.5 * log_std.shape[0] * (1.0 + np.log(2*np.pi)) + log_std.sum()

def diag_gausian_log_density(x, mu, log_std):
	n = x.shape[0]

	std = np.exp(log_std)

	return -0.5*n*np.log(2*np.pi) - 0.5*np.sum(log_std) - ((x - mu)**2/(2* (std**2))).sum()

def vlb_objective_simple(params, y,  mu_y, N_samples=10):
	mean, log_std = params
	weights = sample_weights(params, n_samples)
	entropy = gaussian_entropy(log_std)

	log_likelihood = diag_gausian_log_density(y, mu_y, .1)
	log_prior = diag_gausian_log_density(weights, 0, 1)

	return -entropy - np.mean(log_likelihood + log_prior)

def vlb_objective(params, y, inputs, num_mc_weight_samples, ax):
	""" estimates elbo = -H[q(w))]- E_q(w)[log p(D,w)] """
	mean, log_std = params
	#weights = sample_weights(params, n_samples)
	entropy = gaussian_entropy(log_std)

	#f_bnn = sample_bnn(params, x, n_samples,layer_sizes, act)
   
	log_likelihood_samples = []
	log_prior_samples = []

	for ii in range(num_mc_weight_samples):
		
		y_hat, sampled_weights = model.forward(inputs, num_mc_weight_samples)

		log_likelihood = diag_gausian_log_density(y, y_hat, .1)
		log_prior = diag_gausian_log_density(sampled_weights, 0, 1)
	
		log_likelihood_samples += [log_likelihood]
		log_prior_samples += [log_prior]


		num_data = y_hat.shape[0]
		plot_inputs = np.linspace(-8,8, num=num_data)
		ax.plot(plot_inputs, y_hat.view(-1).numpy())
		ax.set_ylim([-5,5])
		plt.draw()
		plt.pause(1.0/60.0)

	log_likelihood_samples = torch.Tensor(log_likelihood_samples)
	log_prior_samples = torch.Tensor(log_prior_samples)

	print('entropy', entropy.item())
	print('mean log_likelihood', log_likelihood_samples.mean().item())
	print('mean log_prior', log_prior_samples.mean().item())

	loss = - entropy - (log_likelihood_samples + log_prior_samples).mean()

	return loss 

# train & test functions

def train(model, n_data, num_mc_weight_samples = 10):
	model.train()
	
	params = (model.mean, model.log_std)
	inputs, targets = sample_data(n_data)

	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	plt.ion()
	plt.show(block=False)
	

	for input_, target in zip(inputs, targets):
		optimizer.zero_grad()

		#vlb_objective_simple(params, y,  mu_y, N_samples=10)

		loss = vlb_objective(params, target, input_, num_mc_weight_samples, ax)

		loss.backward()
		optimizer.step()

		plt.cla()
		ax.plot(inputs.numpy(), targets.numpy(), 'k.')
		#		#ax.plot(plot_inputs, f_bnn.T, color='r')
		#ax.set_ylim([-5, 5])
		#plt.draw()
		#plt.pause(1.0 / 60.0)

		print('training loss', loss.item())

def test(model, n_data = 20):
	model.eval()

	with torch.no_grad():
		inputs, targets = sample_data(n_data)
		outputs, sampled_weights = model.forward(inputs)

		test_loss = ((outputs - targets)**2).mean()

		print('test_loss', test_loss.item())


# data functions

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


if __name__ == '__main__':
	
	device = torch.device('cpu')
	
	# get the model instance
	arch = [1, 20, 20, 1]
	model = BNN_wrapper(arch).to(device)
	
	## Test
	optimizer = optim.Adam(model.parameters())

	n_data = 50
	num_mc_weight_samples = 50
	train(model, n_data)
	test(model)

	pdb.set_trace()
