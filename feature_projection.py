import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function


# class ProjectionNet(nn.Module):
#     def __init__(self, dim_in, head='mlp', feat_dim=128):
#         super(ProjectionNet, self).__init__()
        
#         self.leaky_relu = nn.LeakyReLU(0.1)
#         self.conv0 = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True)
#         #      self.bn1 = nn.BatchNorm2d(planes)
#         self.conv1 = nn.Conv2d(dim_in, 32, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
#         #      self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
#         if head == 'linear':
#             self.head = nn.Linear(dim_in, feat_dim)
#         elif head == 'mlp':
#             self.head = nn.Sequential(
#                 nn.Linear(1024, 512),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(512, feat_dim)
#             )
#         else:
#             raise NotImplementedError(
#                 'head not supported: {}'.format(head))

#     def forward(self, x):
#         #print(x.shape)
#         out = self.leaky_relu((self.conv0(x)))
#         out = self.leaky_relu((self.conv1(out)))
#         out = self.leaky_relu((self.conv2(out)))
#         out = self.leaky_relu((self.conv3(out)))
#         out = self.leaky_relu((self.conv4(out)))
#         #print(out.shape)
#         # flatten x to (bs, dim_in)
#         x = torch.flatten(out, 1)
#         #print(out.shape)
#         feat = F.normalize(self.head(x), dim=1)
#         return feat, x

class ProjectionNet(nn.Module):
    def __init__(self, dim_in, head='mlp', feat_dim=128):
        super(ProjectionNet, self).__init__()
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv0 = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True)
        #      self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(dim_in, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        #      self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        #print(x.shape)
        out = self.leaky_relu((self.conv0(x)))
        out = self.leaky_relu((self.conv1(out)))
        out = self.leaky_relu((self.conv2(out)))
        out = self.leaky_relu((self.conv3(out)))
        out = self.leaky_relu((self.conv4(out)))

        #print(out.shape)
        # flatten x to (bs, dim_in)
        x = torch.flatten(out, 1)
        #print(out.shape)
        feat = F.normalize(self.head(x), dim=1)
        return feat, x




class Regressornet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Regressornet, self).__init__()
    # Define transposed convolution with padding
    self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    x = self.conv1(x)
    x = nn.functional.relu(x)
    return x
