import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


# # 创建模型实例
# model = CBAM(channels=64, reduction=16)
# import torch
# import torch.nn as nn
# from torch import Tensor
#
#
# class CBAM(nn.Module):
#     def __init__(self, channel, reduction=16, spatial_kernel=7):
#         super(CBAM, self).__init__()
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # shared MLP
#         self.mlp = nn.Sequential(
#             # Conv2d比Linear方便操作
#             # nn.Linear(channel, channel // reduction, bias=False)
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # inplace=True直接替换，节省内存
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel,bias=False)
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
#                               padding=spatial_kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x : Tensor):
#         x = self.max_pool(x)
#         max_out = self.mlp(x)
#         x = self.avg_pool(x)
#         avg_out = self.mlp(x)
#         channel_out = self.sigmoid(max_out + avg_out)
#         x = channel_out * x
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
#         x = spatial_out * x
#         return x