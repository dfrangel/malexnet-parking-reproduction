import torch
import torch.nn as nn

class mAlex(nn.Module):
    def __init__(self):
        super().__init__()

        # First Layer: Conv + ReLU + LRN + MaxPool
        self.layer1 = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=1.0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Second Layer: Conv + ReLU + LRN + MaxPool
        self.layer2 = nn.Sequential (
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=1.0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Third Layer: Conv + ReLU + MaxPool
        self.layer3 = nn.Sequential (
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Fourth Layer: FC + ReLU
        self.layer4 = nn.Sequential (
            nn.Linear(in_features=30*3*3, out_features=48),
            nn.ReLU(inplace=True)
        )
        # Fifth Layer: FC
        self.layer5 = nn.Sequential (
            nn.Linear(in_features=48, out_features=2)
        )

    def forward (self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
