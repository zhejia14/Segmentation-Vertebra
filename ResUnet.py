import torch
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import BatchNorm2d
from torch.nn import Sequential
from torchvision.transforms import CenterCrop
from torch.nn import functional as F


class DoubleConv(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class enBlock(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.double_conv = DoubleConv(inChannels, outChannels)
        self.downsample = Conv2d(inChannels, outChannels, kernel_size=1, stride=1)
        self.pool = MaxPool2d(2)

    def forward(self, x):
        identity = self.downsample(x)
        conv_out = self.double_conv(x)
        pool = self.pool(identity + conv_out)
        return pool


class Bridge(Module):
    def __init__(self, inChannels):
        super().__init__()
        self.double_conv1 = DoubleConv(inChannels, inChannels * 2)
        self.downsample = Conv2d(inChannels, inChannels * 2, kernel_size=1, stride=1)
        self.double_conv2 = DoubleConv(inChannels * 2, inChannels)
        self.upsample = ConvTranspose2d(inChannels * 2, inChannels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.downsample(x)
        conv_out1 = self.double_conv1(x)
        conv_out2 = self.double_conv2(identity + conv_out1)
        identity = self.upsample(conv_out1)
        return identity + conv_out2


class deBlock(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.double_conv = DoubleConv(inChannels, outChannels)
        self.upsample = ConvTranspose2d(inChannels, outChannels, kernel_size=1, stride=1)
        
    def forward(self, x):
        identity = self.upsample(x)
        conv_out = self.double_conv(x)
        return identity + conv_out


class Encoder(Module):
    def __init__(self, channels=(1, 32, 64, 128)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [enBlock(channels[i], channels[i + 1])for i in range(len(channels) - 1)])

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        blockOutputs.append(x)
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            # return the list containing the intermediate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels=(128, 64, 32)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [deBlock(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures


class Res_UNet(Module):
    def __init__(self, encChannels=(1, 64, 128, 256, 512),
                 decChannels=(512, 256, 128, 64),
                 nbClasses=1, retainDim=True,
                 outSize=(1200, 500)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.bridge = Bridge(decChannels[0])
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize
        

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        encFeatures[-1] = self.bridge(encFeatures[-1])
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize, mode='nearest')
        # return the segmentation map
        return torch.sigmoid(map)
