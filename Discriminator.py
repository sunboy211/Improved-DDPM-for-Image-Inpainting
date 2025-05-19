import torch.nn as nn
'''
refer to PatchGAN Discriminator: "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py"
'''
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(PatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
