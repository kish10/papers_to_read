import numpy as np
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms


HOME = os.getcwd().split('paper_reviews')[0]
DATA_DIR = os.path.join(HOME, 'data')
RESULTS_DIR = os.path.join(HOME, 'results','weight_uncertainty_in_nns')

def get_data_loaders(use_cuda = False, batch_size = 128):
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

class NN(nn.Module):
    """Neural Network to recognize digits in MNIST images"""

    def __init__(self):
        super(NN, self).__init__()
        
        # specify the linear components of the NN layers
        self.linL1 = nn.Linear(784, 20) # 784 pixels per image
        self.linL2 = nn.Linear(20, 10) # output 10 probabilites for each digit



    def forward(self, images):
        # reshape images into tensor with 784 collumns
        images = images.view(-1, 784)  

        mid_layer = F.relu(self.linL1(images))
        log_digit_probablities = F.log_softmax(self.linL2(mid_layer), dim=1)

        return log_digit_probablities


def loss_function(log_digit_probablities, digits):
    """Use a negative likelihood loss function to find the MLE for the weights"""

    neg_log_likelihood_loss = F.nll_loss(log_digit_probablities, digits)

    return neg_log_likelihood_loss


def train(epoch, model, train_loader, max_observations = 10**3):
    model.train() # sets the model in "training" mode
    train_loss = 0

    for batch_index, (images, digits) in enumerate(train_loader):
        if batch_index > max_observations:
            break

        images = images.to(device) # why is this required ?

        # clear computed values based on previous batch
        optimizer.zero_grad()   

        # input data and compute the loss using the output from the model
        log_digit_probablities = model.forward(images)
        loss = loss_function(log_digit_probablities, digits)

        # compute gradients
        loss.backward()
        train_loss += loss.item()
        optimizer.step() 
        # - how does it update the weight estimates?
        # - how does it use information from loss.backward()

        # print training information
        log_interval = 10
        if batch_index % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            .format(
                epoch, batch_index * len(images), 
                len(train_loader.dataset),
                100. * batch_index / len(train_loader), 
                loss.item() / len(images)
                )
            )
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch, model, test_loader, max_test_comparisons_printed = 10):
    model.eval()
    test_loss = 0

    with torch.no_grad():
    # "Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward()"
        for batch_index, (images, digits) in enumerate(test_loader):
            images = images.to(device)
            log_digit_probablities = model(images)

            test_loss += loss_function(log_digit_probablities, digits)

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

    device = torch.device('cpu')

    # get the model instance
    model = NN().to(device)
    optimizer = optim.Adam(model.parameters())

    # get datasets
    train_loader, test_loader = get_data_loaders()

    # train & test model
    epochs = 1

    for epoch in range(epochs + 1):
        train(epoch, model, train_loader)
        test(epoch, model, test_loader)