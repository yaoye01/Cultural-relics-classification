import torch
import torch.nn as nn


class Block(nn.Module):
    """
    param : in_channel--输入通道数
            mid_channel -- 中间经历的通道数
            out_channel -- ResNet部分使用的通道数（sum操作，这部分输出仍然是out_channel个通道）
            dense_channel -- DenseNet部分使用的通道数（concat操作，这部分输出是2*dense_channel个通道）
            groups -- conv2中的分组卷积参数
            is_shortcut -- ResNet前是否进行shortcut操作
    """

    def __init__(self, in_channel, mid_channel, out_channel, dense_channel, stride, groups, is_shortcut=False):
        super(Block, self).__init__()

        self.is_shortcut = is_shortcut
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel + dense_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel + dense_channel)
        )

        if self.is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel + dense_channel, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel + dense_channel)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        a = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_shortcut:
            a = self.shortcut(a)

        # a[:, :self.out_channel, :, :]+x[:, :self.out_channel, :, :]是使用ResNet的方法，即采用sum的方式将特征图进行求和，
        # 通道数不变，都是out_channel个通道
        # a[:, self.out_channel:, :, :], x[:, self.out_channel:, :, :]]是使用DenseNet的方法，
        # 即采用concat的方式将特征图在channel维度上直接进行叠加，通道数加倍，即2*dense_channel
        # 注意最终是将out_channel个通道的特征（ResNet方式）与2*dense_channel个通道特征（DenseNet方式）进行叠加，因此最终通道数为out_channel+2*dense_channel
        x = torch.cat([a[:, :self.out_channel, :, :] + x[:, :self.out_channel, :, :], a[:, self.out_channel:, :, :],
                       x[:, self.out_channel:, :, :]], dim=1)
        x = self.relu(x)

        return x


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()

        self.group = cfg['group']
        self.in_channel = cfg['in_channel']
        mid_channels = cfg['mid_channels']
        out_channels = cfg['out_channels']
        dense_channels = cfg['dense_channels']
        num = cfg['num']

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, 7, stride=2, padding=3, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.conv2 = self._make_layers(mid_channels[0], out_channels[0], dense_channels[0], num[0], stride=1)
        self.conv3 = self._make_layers(mid_channels[1], out_channels[1], dense_channels[1], num[1], stride=2)
        self.conv4 = self._make_layers(mid_channels[2], out_channels[2], dense_channels[2], num[2], stride=2)
        self.conv5 = self._make_layers(mid_channels[3], out_channels[3], dense_channels[3], num[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg['out_channels'][3] + (num[3] + 1) * cfg['dense_channels'][3], cfg['classes'])  # fc层需要计算

    def _make_layers(self, mid_channel, out_channel, dense_channel, num, stride):
        layers = []
        # is_shortcut=True表示进行shortcut操作，则将浅层的特征进行一次卷积后与进行第三次卷积的特征图相加（ResNet方式）和concat(DeseNet方式)操作
        # 第一次使用Block可以满足浅层特征的利用，后续重复的Block则不需要线层特征，因此后续的Block的is_shortcut=False(默认值)
        layers.append(Block(self.in_channel, mid_channel, out_channel, dense_channel, stride=stride, groups=self.group,
                            is_shortcut=True))
        self.in_channel = out_channel + dense_channel * 2
        for i in range(1, num):
            layers.append(Block(self.in_channel, mid_channel, out_channel, dense_channel, stride=1, groups=self.group))
            # 由于Block包含DenseNet在叠加特征图，所以第一次是2倍dense_channel，后面每次都会多出1倍dense_channel
            self.in_channel += dense_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
