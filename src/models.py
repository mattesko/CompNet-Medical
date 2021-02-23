import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models


class UNet(nn.Module):

    def __init__(self, dice=False, in_channels=1, pretrained=False, 
                 num_classes=2, bias=None):

        super().__init__()
        
        if pretrained:
            encoder = models.vgg13(pretrained=True).features
            self.conv1_input = encoder[0] # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv1 =       encoder[2] # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2_input = encoder[5] # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 =       encoder[7] # Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv3_input = encoder[10] # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv3 =       encoder[12] # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv4_input = encoder[15] # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv4 =       encoder[17] # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

#             self.conv6 =       encoder[17] # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            self.conv1_input =      nn.Conv2d(in_channels, 64, 3, padding=1)
            self.conv1 =            nn.Conv2d(64, 64, 3, padding=1)
            self.conv2_input =      nn.Conv2d(64, 128, 3, padding=1)
            self.conv2 =            nn.Conv2d(128, 128, 3, padding=1)
            self.conv3_input =      nn.Conv2d(128, 256, 3, padding=1)
            self.conv3 =            nn.Conv2d(256, 256, 3, padding=1)
            self.conv4_input =      nn.Conv2d(256, 512, 3, padding=1)
            self.conv4 =            nn.Conv2d(512, 512, 3, padding=1)
            
#             self.conv6 =            nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv5_input =      nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5 =            nn.Conv2d(1024, 1024, 3, padding=1)

        self.conv6_up =         nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6_input =      nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6 =            nn.Conv2d(512, 512, 3, padding=1)
        self.conv7_up =         nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7_input =      nn.Conv2d(512, 256, 3, padding=1)
        self.conv7 =            nn.Conv2d(256, 256, 3, padding=1)
        self.conv8_up =         nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8_input =      nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 =            nn.Conv2d(128, 128, 3, padding=1)
        self.conv9_up =         nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9_input =      nn.Conv2d(128, 64, 3, padding=1)
        self.conv9 =            nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_output =     nn.Conv2d(64, num_classes, 1)

        if dice:
            self.final =        nn.Softmax(dim=1)
        else:
            self.final =        nn.LogSoftmax(dim=1)
            
        if bias:
            nn.init.constant_(self.final, bias)

    def switch(self, dice):

        if dice:
            self.final =        nn.Softmax(dim=1)
        else:
            self.final =        nn.LogSoftmax(dim=1)

    def forward(self, x):

        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        layer5 = F.max_pool2d(layer4, 2)
        layer5 = F.relu(self.conv5_input(layer5))
        layer5 = F.relu(self.conv5(layer5))

        layer6 = F.relu(self.conv6_up(layer5))
        layer6 = torch.cat((layer4, layer6), 1)
        layer6 = F.relu(self.conv6_input(layer6))
        layer6 = F.relu(self.conv6(layer6))

        layer7 = F.relu(self.conv7_up(layer6))
        layer7 = torch.cat((layer3, layer7), 1)
        layer7 = F.relu(self.conv7_input(layer7))
        layer7 = F.relu(self.conv7(layer7))

        layer8 = F.relu(self.conv8_up(layer7))
        layer8 = torch.cat((layer2, layer8), 1)
        layer8 = F.relu(self.conv8_input(layer8))
        layer8 = F.relu(self.conv8(layer8))

        layer9 = F.relu(self.conv9_up(layer8))
        layer9 = torch.cat((layer1, layer9), 1)
        layer9 = F.relu(self.conv9_input(layer9))
        layer9 = F.relu(self.conv9(layer9))
#         layer9 = self.final(self.conv9_output(layer9))
        layer9 = self.conv9_output(layer9)

        return layer9
    
    def get_bottleneck_feature_map(self, x):
        layer1 = F.relu(self.conv1_input(x))
        layer1 = F.relu(self.conv1(layer1))

        layer2 = F.max_pool2d(layer1, 2)
        layer2 = F.relu(self.conv2_input(layer2))
        layer2 = F.relu(self.conv2(layer2))

        layer3 = F.max_pool2d(layer2, 2)
        layer3 = F.relu(self.conv3_input(layer3))
        layer3 = F.relu(self.conv3(layer3))

        layer4 = F.max_pool2d(layer3, 2)
        layer4 = F.relu(self.conv4_input(layer4))
        layer4 = F.relu(self.conv4(layer4))

        layer5 = F.max_pool2d(layer4, 2)
        layer5 = F.relu(self.conv5_input(layer5))
        layer5 = F.relu(self.conv5(layer5))
        
        return layer5
    
    def get_features(self):
        return nn.Sequential(self.conv1_input,
                             nn.ReLU(),
                             self.conv1,
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             self.conv2_input,
                             nn.ReLU(),
                             self.conv2,
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             self.conv3_input,
                             nn.ReLU(),
                             self.conv3,
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             self.conv4_input,
                             nn.ReLU(),
                             self.conv4,
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             self.conv5_input,
                             nn.ReLU(),
                             self.conv5,
                             nn.ReLU(),
                             self.conv6_up,
                             nn.ReLU(),
                             self.conv6_input,
                             self.conv6,
                             self.conv7_up,
                             self.conv7_input,
                             self.conv7,
                             self.conv8_up,
                             self.conv8_input,
                             self.conv8,
                             self.conv9_up,
                             self.conv9_input,
                             self.conv9,
                             self.conv9_output,
                             self.final
        )
    
def set_weights_to_ones(model):
    for i, layer in enumerate(model):
        if type(layer) == nn.Conv2d:
            model[i].weight = nn.Parameter(torch.ones(layer.weight.shape))
    return model
