import pdb


import numpy as np
import os
import torch
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torchvision import datasets, transforms


HOME = os.getcwd().split('paper_reviews')[0]
DATA_DIR = os.path.join(HOME, 'data')
RESULTS_DIR = os.path.join(HOME, 'results','weight_uncertainty_in_nns')

def get_data_loaders(use_cuda = False, batch_size = 2): # size = 128
    """Returns pytorch train & test data loaders"""

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        dataset = datasets.MNIST(
            DATA_DIR, 
            train = True,
            download = True,
            transform = transforms.ToTensor() # can this not be in function call version ?
        ),
        batch_size = batch_size,
        shuffle = True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = datasets.MNIST(
            DATA_DIR, 
            train = False,
            transform = transforms.ToTensor()),
        batch_size = batch_size,
        shuffle = True,
        **kwargs
    )

    return train_loader, test_loader

class BNN(nn.Module):

    def __init__(self):
        super(BNN, self).__init__()

        # specify the linear components in the different NN layers
        self.linL1_weight_shape = [784, 20]
        self.linL2_weight_shape = [20, 10]

        self.linL1 = nn.Linear(*self.linL1_weight_shape)
        self.linL2 = nn.Linear(*self.linL2_weight_shape)

        #  get number of parameters in the different layers
        self.number_of_linL1_weights = np.prod([*self.linL1.weight.shape])
        number_of_linL2_weights = np.prod([*self.linL2.weight.shape])
        self.total_number_of_weights = (
            self.number_of_linL1_weights 
            + number_of_linL2_weights
            )

        # initialize the latent parameters specified in the variational posterior
        initial_mus = torch.randn(self.total_number_of_weights, 1)
        initial_rhos = (torch
                        .ones(self.total_number_of_weights, 1)
                        .exponential_()
                       )
        self.mu = nn.Parameter(initial_mus)
        self.rho = nn.Parameter(initial_rhos)

    
    def sample_weights(self):
        # Note that Pytorch handles "weights" & biases of linear parts differently.
        # TODO should try to see if it's possible to reduce the dimension of mu & stdev/rho

        stdev = (
            torch.ones(self.total_number_of_weights, 1) 
            + self.rho.exp()
            ).log()

        epsilons = torch.randn(self.total_number_of_weights, 1)
        weights = self.mu + stdev*epsilons

        return weights

    def make_weights_usable(self, weights):
        linL1_weights = nn.Parameter(
            weights[:self.number_of_linL1_weights]
            .view(*self.linL1_weight_shape[::-1])
        )

        linL2_weights = nn.Parameter(
            weights[self.number_of_linL1_weights:]
            .view(*self.linL2_weight_shape[::-1])
        )

        return [linL1_weights, linL2_weights]

    def forward(self, images):
        # reshape images into tensor with 784 collumns
        images = images.view(-1, 784)  

        # get sampled weights for the NN
        weights = self.sample_weights()

        # set weights
        linear_weights = self.make_weights_usable(weights)

        self.linL1.weight = linear_weights[0]
        self.linL2.weight = linear_weights[1]

        # compose the model
        mid_layer = F.relu(self.linL1(images))
        log_digit_probablities = F.log_softmax(self.linL2(mid_layer), dim=1)

        return weights, log_digit_probablities #, self.mu, self.rho


def loss_function(
    mc_weights_sample, 
    mu_vec, rho_vec,
    mc_log_digit_probablities_samples, 
    digits
    ):
    """Use a negative likelihood loss function to find the MLE for the weights"""

    # TODO add the KL term b/w varitional posterior & prior

    # Testist variations of the model
    # TODO extend loss from one to multiple monte carlo sample of weights
    # TODO test claim in paper that closed form is equivalent to MC sampled sum
    # TODO test claim that non closed form priors are be better in practice
    # TODO test above with different data.
    #       - maybe can test this in conjuction with other deeper investigations
    #           like Daniel's paper.

    neg_log_likelihood_loss_sum = 0
    var_pos_log_likelihood_sum = 0
    prior_log_likelihood_sum = 0
    kl_loss_sum = 0

    # get sum of the negative likelihood loss
    for log_digit_probablities in mc_log_digit_probablities_samples:
        neg_log_likelihood_loss = F.nll_loss(log_digit_probablities, digits, reduction='sum')
        neg_log_likelihood_loss_sum += neg_log_likelihood_loss


    print("sample's log_digit_probablities", log_digit_probablities)
    print("sample's neg_log_likelihood_loss", np.round(neg_log_likelihood_loss.item(), 2))

    pdb.set_trace()

    # TODO finish KL part of loss.

    # get sum of the prior & posterior densities
    stdev_vec = (1 + rho_vec.exp()).log()
    variance_vec = stdev_vec**2

    # TODO finish the summed loss
    for weights in mc_weights_sample:
        log_prior_densities = Normal(0, 1).log_prob(weights)

        log_var_post_densities = [
            [Normal(mu, variance).log_prob(weight)] 
            for weight, mu, variance 
            in zip(weights, mu_vec, variance_vec)
        ]
        log_var_post_densities = torch.Tensor(log_var_post_densities)

        print('samples KL loss', 
             torch.sum(log_var_post_densities - log_prior_densities).item()
             )
        print('----')

        kl_loss_sum += torch.sum(log_var_post_densities - log_prior_densities) 

        #var_pos_log_likelihood_sum += torch.sum(log_var_post_densities)
        #prior_log_likelihood_sum += torch.sum(log_prior_densities)


    loss = (
        #var_pos_log_likelihood_sum 
        #- prior_log_likelihood_sum
        kl_loss_sum
        + neg_log_likelihood_loss_sum 
    )

    print(
    #    'var_pos_log_likelihood_sum', 
    #     np.round(var_pos_log_likelihood_sum,2),
    #      'prior_log_likelihood_sum',
    #      np.round(prior_log_likelihood_sum,2),
          'neg_log_likelihood_loss_sum',
          np.round(neg_log_likelihood_loss_sum.item(),2)
         )

    #pdb.set_trace()

    return loss


#device = torch.device('cpu')
#model = BNN().to(device)
#optimizer = optim.Adam(model.parameters())


def train(
    epoch, 
    model, 
    train_loader, max_observations = 10**1,
    number_of_monte_carlo_samples = 1
    ):
    model.train()
    train_loss = 0

    for batch_index, (images, digits) in enumerate(train_loader):
        if batch_index > max_observations:
            break
        

        images = images.to(device)

        optimizer.zero_grad()

        # get Monte Carlo samples
        mc_weights_sample = []
        mc_log_digit_probablities_sample = []

        for ii in range(number_of_monte_carlo_samples):
            weights, log_digit_probablities = model(images)

            mc_weights_sample += [weights]
            mc_log_digit_probablities_sample += [log_digit_probablities]

        # calculate loss
        mu = model.mu
        rho = model.rho

        train_loss += loss_function(
            mc_weights_sample,
            mu, rho,
            mc_log_digit_probablities_sample,
            digits
            )

        # print training information     
        log_interval = 10     
        if batch_index % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                .format(
                    epoch, batch_index * len(images),
                    len(train_loader.dataset),
                    100. * batch_index / len(train_loader),
                    train_loss.item() / len(images)
                )
            )


    # TODO add print statements with the loss

# TODO add test function

def test(epoch, model, test_loader, max_test_comparisons_printed = 10):
    model.eval()
    test_loss = 0

    with torch.no_grad():
    # "Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward()"
        for batch_index, (images, digits) in enumerate(test_loader):
            images = images.to(device)
            log_digit_probablities = model(images)

        
            # get Monte Carlo samples
            mc_weights_sample = []
            mc_log_digit_probablities_sample = []


            number_of_monte_carlo_samples = 1
            for ii in range(number_of_monte_carlo_samples):
                weights, log_digit_probablities = model(images)

                mc_weights_sample += [weights]
                mc_log_digit_probablities_sample += [log_digit_probablities]

            # calculate loss
            mu = model.mu
            rho = model.rho

            test_loss += loss_function(
                mc_weights_sample,
                mu, rho,
                mc_log_digit_probablities_sample,
                digits
                )

            #test_loss += loss_function(log_digit_probablities, digits)

            if batch_index < 2:
                # print digit and digit probabilities
                digits_and_logprobs = zip(digits, log_digit_probablities)
                for i, (digit, dig_probs) in enumerate(digits_and_logprobs):
                    if i > max_test_comparisons_printed:
                        break

                    prediction = dig_probs.argmax()
                    print( 'Digit: {} ---- Prediction: {}'
                        .format(digit, prediction)
                    )

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':

    BATCH_SIZE = 1
    
    device = torch.device('cpu')

    # get the model instance
    model = BNN().to(device)
    optimizer = optim.Adam(model.parameters())

    # get datasets
    train_loader, test_loader = get_data_loaders(batch_size = BATCH_SIZE)

    # train & test model
    epochs = 0

    for epoch in range(epochs + 1):
        train(epoch, model, train_loader)
        test(epoch, model, test_loader)
