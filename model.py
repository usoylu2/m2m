import torch.nn as nn
import torch
import torchvision.models as models

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ResNet Model
class ResNet(nn.Module):
    def __init__(self):
        """
        Constructor for the ResNet model.

        Initializes the layers and components of the ResNet model.

        - Creates a 2D convolutional layer with 1 input channel, 64 output channels, a 7x7 kernel,
          2x2 stride, and 3x3 padding.
        - Loads a pre-trained ResNet-50 model and extracts its feature extraction layers, excluding
          the final classification layer.
        - Adds a linear (fully connected) layer with 2048 input features and 1 output logit for classification.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.premodel = nn.Sequential(*list(models.resnet50(pretrained=True).children())[1:-1])
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        - Applies the initial convolutional layer to the input.
        - Passes the result through the feature extraction layers obtained from the pre-trained ResNet model.
        - Flattens the output tensor.
        - Applies the fully connected layer for classification.

        Args:
        x (Tensor): Input data with shape (batch_size, 1, height, width).

        Returns:
        x (Tensor): Output tensor with shape (batch_size, 1) representing classification logits.

        """
        x = self.conv1(x)
        x = self.premodel(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# DenseNet Model
class DenseNet(nn.Module):
    def __init__(self):
        """
        Constructor for the DenseNet model.

        Initializes the layers and components of the DenseNet model.

        - Creates a 2D convolutional layer with 1 input channel, 64 output channels, a 7x7 kernel,
          2x2 stride, and 5x5 padding.
        - Loads a pre-trained DenseNet-201 model and extracts its feature extraction layers.
        - Adds a linear (fully connected) layer with 11520 input features and 1 output logit for classification.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(5, 5), bias=False)
        self.premodel = nn.Sequential(*list(list(models.densenet201(pretrained=True).children())[0].children())[1:])
        self.fc = nn.Linear(11520, 1)

    def forward(self, x):
        """
        Forward pass of the DenseNet model.

        - Applies the initial convolutional layer to the input.
        - Passes the result through the feature extraction layers obtained from the pre-trained DenseNet model.
        - Flattens the output tensor.
        - Applies the fully connected layer for regression.

        Args:
        x (Tensor): Input data with shape (batch_size, 1, height, width).

        Returns:
        x (Tensor): Output tensor with shape (batch_size, 1) representing classification logits.

        """
        x = self.conv1(x)
        x = self.premodel(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
