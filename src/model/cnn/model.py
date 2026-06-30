import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self):
        self.kernel = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=14, stride=14)
        self.
