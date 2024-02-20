import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes # N
        self.samples_per_class = samples_per_class # K

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images (N, 784) #characters, actual image data 
            labels: [B, K+1, N, N] ground truth labels (N,N) #characters, 1-hot-encoded label, 1 for each N character
        Returns:
            [B, K+1, N, N] predictions # prediction for each character
        """ 
        #############################
        ### START CODE HERE ###
        B, K, N, _ = input_images.size()

        x = torch.cat((input_images, input_labels), dim=3)
        x[:, -1, :, 784:] = 0
        x = x.reshape(x.shape[0], -1, x.shape[3])
        x = x.to(torch.float32)
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)

        x = x.reshape(x.shape[0], K, N , -1)

        return x
        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ###


        # print("preds.shape: ", preds.shape)
        # print("labels.shape: ", labels.shape)

        
        # Extract the predictions for the test images
        test_preds = preds[:, -1]  # Shape: [B, N, N]
        # Extract the labels for the test images
        test_labels = labels[:, -1]  # Shape: [B, N, N]

        # Compute cross-entropy loss
        loss = F.cross_entropy(test_preds, test_labels)


        return loss
        ### END CODE HERE ###

