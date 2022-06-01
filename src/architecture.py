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
  "identity": lambda x:x,
  "prelu": nn.PReLU()
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
   
class Generator(nn.Module):
    def __init__(self, params):
        upsample_block_num = params["n_aug_blocks"]
        n_res_blocks = params["n_res_blocks"]

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64, params) for _ in range(n_res_blocks)])
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        upsample_blocks = [UpsampleBLock(64, 2, params) for _ in range(upsample_block_num)]
        upsample_blocks.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        self.F = FUNCS[params["out_TF"]]

    def forward(self, x):
        block1 = self.block1(x)
        out = self.res_blocks(block1)
        out = self.block2(out)
        out = self.upsample_blocks(block1 + out)

        return (self.F(out) + 1) / 2
 
class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.F =FUNCS[params["disc_TF"]]
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
        return self.F(self.net(x).view(batch_size))

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
    def __init__(self, channels, params):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.F = FUNCS[params["res_TF"]]
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.F(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, params):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.F = FUNCS[params["up_TF"]]

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

