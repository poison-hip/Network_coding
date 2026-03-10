import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from mobilenetv2 import mobilenetv2


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__()
        if downsample_factor not in [8, 16]:
            raise ValueError("downsample_factor must be 8 or 16")

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim ,kernel_size=1, stride=1, padding=0, dilation=rate, bias=False),
            nn.BatchNorm2d(out_dim, momentum=bn_mom),
            nn.ReLU6(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=6*rate, dilation=6*rate, bias=False),
            nn.BatchNorm2d(out_dim, momentum=bn_mom),
            nn.ReLU6(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=12*rate, dilation=12*rate, bias=False),
            nn.BatchNorm2d(out_dim, momentum=bn_mom),
            nn.ReLU6(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=18*rate, dilation=18*rate, bias=False),
            nn.BatchNorm2d(out_dim, momentum=bn_mom),
            nn.ReLU6(inplace=True)
        )

        self.branch5_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch5_bn = nn.BatchNorm2d(out_dim, momentum=bn_mom)
        self.branch5_relu = nn.ReLU6(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_dim*5, out_dim, 1, 1,padding=0,bias=True),
            nn.BatchNorm2d(out_dim, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError(
                "Unsupported backbone - '{}', only mobilenet is supported.".format(backbone)
            )

        self.aspp = ASPP(in_dim=in_channels, out_dim=256, rate=16 // downsample_factor)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(
            x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x

    
