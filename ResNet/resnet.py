import torch 
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, input_channel, nums_channel, conv_1x1 = False, strides=1,):
        super(Residual,self).__init__()
        self.conv1 = nn.Conv2d(input_channel, nums_channel, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(nums_channel, nums_channel, kernel_size=3, padding=1)
        if conv_1x1:
            self.conv3 = nn.Conv2d(input_channel, nums_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(nums_channel)
        self.bn2 = nn.BatchNorm2d(nums_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self ,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, padding=1, stride=2),
                   nn.BatchNorm2d(64),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                   )

def resnet_block(input_channel, num_channel, num_residual, first_block=False):
    bk = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            bk.append(*Residual(input_channel, num_channel, 
                                strides=2, padding = 1))
        else:
            bk.append(*Residual(input_channel, num_channel))
        
    return bk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b2, b3, b4, b5, nn.AdaptiveAvgPool2d(1,1),
                  nn.Flatten(), nn.Linear(512, 10))

