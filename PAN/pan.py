import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyBackbone(nn.Module):
    """
    输出 3 个不同尺度特征:
    C3: 1/8
    C4: 1/16
    C5: 1/32
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1),   # 1/2
            ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1),  # 1/4
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1),  # 1/8
            ConvBNReLU(128, 128),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1),  # 1/16
            ConvBNReLU(256, 256),
        )
        self.stage5 = nn.Sequential(
            ConvBNReLU(256, 512, kernel_size=3, stride=2, padding=1),  # 1/32
            ConvBNReLU(512, 512),
        )

    def forward(self, x):
        x = self.stem(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class FPN(nn.Module):
    """
    标准 FPN（自顶向下）:
    C5 -> P5
    C4 + up(P5) -> P4
    C3 + up(P4) -> P3
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list]
        )

    def forward(self, features):
        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled

        p3, p4, p5 = [out_conv(f) for out_conv, f in zip(self.output_convs, laterals)]
        return p3, p4, p5


class PAN(nn.Module):
    """
    PAN 路径聚合:
    先做 FPN(自顶向下) 得到 P3/P4/P5，
    再做自底向上聚合，增强高层的定位细节。
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.fpn = FPN(in_channels_list=in_channels_list, out_channels=out_channels)

        # bottom-up: N3 -> N4 -> N5
        self.downsample_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.downsample_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.out_n4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_n5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        # top-down FPN
        p3, p4, p5 = self.fpn(features)

        # bottom-up PAN
        n3 = p3
        n4 = self.out_n4(p4 + self.downsample_p3(n3))
        n5 = self.out_n5(p5 + self.downsample_p4(n4))

        # 输出聚合后的多尺度特征
        return n3, n4, n5


class PANModel(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.backbone = TinyBackbone()
        self.pan = PAN(in_channels_list=[128, 256, 512], out_channels=out_channels)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        n3, n4, n5 = self.pan([c3, c4, c5])
        return {"N3": n3, "N4": n4, "N5": n5}


if __name__ == "__main__":
    model = PANModel(out_channels=256)
    x = torch.randn(1, 3, 256, 256)
    outputs = model(x)
    for name, feat in outputs.items():
        print(f"{name}: {feat.shape}")
