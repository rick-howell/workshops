# This should be the main CNN class

import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # TODO: Define an activation function


        # TODO: Define a subsample function


        # TODO: Define 2 convolutional layers


        # TODO: Define 1 fully connected layer



    def forward(self, x):

        # TODO: Implement the forward pass
        # The convolutional layers should look like this:
        # - convolve
        # - apply the activation function
        # - apply the subsample function

        # TODO: First Convolutional Layer


        # TODO: Second Convolutional Layer


        # Flatten the output for the fully connected layer
        x = x.view(-1, 16 * 4 * 4)

        # TODO: Fully connected layer


        return x