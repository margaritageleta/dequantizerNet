import warnings
import torch
import math
from torch import nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

warnings.filterwarnings("ignore")

FUNCS = {
  "tanh":nn.Tanh(), 
  "relu":nn.ReLU(), 
  "leakyRelu":nn.LeakyReLU(), 
  "gelu":nn.GELU(),
  "sigmoid": nn.Sigmoid(),
  "identity": lambda x:x
}

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, params):                                                                                                           
        super().__init__()
        self.upsampling_method = params['upsampling_method']
        if self.upsampling_method == 'deconvolution':
            self.upsample1 = nn.ConvTranspose2d(input_channels, input_channels*2, 5, stride=2, padding=2, output_padding=1)
            self.conv1 = nn.Conv2d(input_channels*2,input_channels*2,3,stride=1, padding=1)
            self.conv2 = nn.Conv2d(input_channels*2,output_channels,3,stride=2, padding=1)
        elif self.upsampling_method == 'bilinear':
            self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv1 = nn.Conv2d(input_channels,input_channels*2,3,stride=1, padding=1)
            self.conv2 = nn.Conv2d(input_channels*2,output_channels,3,stride=2, padding=1)
        else: raise Exception('Unknown upsampling method.')
        self.F = FUNCS[params["conv_TF"]]
        
    def forward(self, x):
        print('\t\t CONV BLOCK:')
        c = x.size(1)
        print(f'Size is {c}')
        x = self.upsample1(x)
        #print(f'After upsample {x.shape}')
        #if self.upsampling_method != 'deconvolution':
        #    o_channels = int(c*2)// c
        #    print(f'O channels: {o_channels}')
        #    x = torch.cat([x for _ in range(o_channels)], axis=1)
        #    print(x.shape)
        print(f'\t\t Module transpose: {x.shape}')
        x = self.F(x)
        x = self.conv1(x)
        # print(f'\t\t Module conv1: {x.shape}')
        x = self.F(x)
        x = self.conv2(x)
        # print(f'\t\t Module conv2: {x.shape}')
        return x

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        # print('\t RESIDUAL BLOCK:')
        # print(f'\t Module inputs: {inputs.shape}')
        outputs = self.module(inputs)
        # print(f'\t Module outputs: {outputs.shape}')
        return torch.cat([self.module(inputs),inputs], dim=1)

class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.pixel_unshuffle = PixelUnshuffle(2)

        self.F = FUNCS[params["block_TF"]]
        self.OUT_F = FUNCS[params["out_TF"]]

        input_channels = 1

        blocks = []
        #AUGMENTING BLOCKS
        for i in range(params["n_aug_blocks"]):
            blocks.append(Residual(ConvBlock(input_channels,input_channels,params)))
            input_channels *= 2
            blocks.append(self.F)
        #REDUCTION BLOCKS
        while input_channels > 1:
            blocks.append(ConvBlock(input_channels,input_channels//2,params))
            input_channels//=2
            if input_channels > 1:
                blocks.append(self.F)
            else:
                blocks.append(self.OUT_F)
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # print(x.shape)
        x = self.pixel_shuffle(x)
        # print(x.shape)
        x = self.blocks(x)
        # print('Blocks:', x.shape)
        x = self.pixel_unshuffle(x)
        # print(x.shape)
        x = torch.narrow(x, 1, 0, 3)
        # print(x.shape)
        return x
    
class Generator2(nn.Module):
    def __init__(self, params):
        upsample_block_num = params["n_aug_blocks"]

        super(Generator2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2
 
class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

class ContentLoss(nn.Module):
    

    def __init__(self, params) -> None:
        super(ContentLoss, self).__init__()

        self.feature_name = params["feature_name"]
        model = models.vgg19(True)
        self.feature_extractor = create_feature_extractor(model, [params["feature_name"]])
        
        self.feature_extractor.eval()

        self.normalize = transforms.Normalize(params["feature_mean"], params["feature_std"])

        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_name]
        hr_feature = self.feature_extractor(hr_tensor)[self.feature_name]

        content_loss = F.mse_loss(sr_tensor, hr_tensor)
        content_loss = F.mse_loss(sr_feature, hr_feature)

        return content_loss

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

