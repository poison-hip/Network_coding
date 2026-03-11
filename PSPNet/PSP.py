import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class PSPModule(nn.Module):

    def __init__(self, in_channels, out_channels, bin_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, size) for size in bin_sizes]
        )
        concat_channels = in_channels + len(bin_sizes) * out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    @staticmethod
    def _make_stage(in_channels, out_channels, bin_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(bin_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature):
        h, w = feature.shape[2], feature.shape[3]
        pyramids = [feature]
        pyramids.extend(
            F.interpolate(stage(feature), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        )
        return self.bottleneck(torch.cat(pyramids, dim=1))


class PSPHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class PSPNet(nn.Module):

    def __init__(self, num_classes, pretrained_backbone=False, use_aux=True):
        super().__init__()
        backbone = resnet50(weights=None if not pretrained_backbone else "DEFAULT")

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.ppm = PSPModule(in_channels=2048, out_channels=512, bin_sizes=(1, 2, 3, 6))
        self.cls_head = PSPHead(in_channels=512, mid_channels=512, num_classes=num_classes, dropout=0.1)

        self.use_aux = use_aux
        if use_aux:
            self.aux_head = PSPHead(in_channels=1024, mid_channels=256, num_classes=num_classes, dropout=0.1)
        else:
            self.aux_head = None

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        aux_feature = self.layer3(x)
        main_feature = self.layer4(aux_feature)

        main_logits = self.cls_head(self.ppm(main_feature))
        main_logits = F.interpolate(
            main_logits, size=(input_h, input_w), mode="bilinear", align_corners=False
        )

        if self.training and self.use_aux:
            aux_logits = self.aux_head(aux_feature)
            aux_logits = F.interpolate(
                aux_logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
            return main_logits, aux_logits
        return main_logits


if __name__ == "__main__":
    model = PSPNet(num_classes=19, pretrained_backbone=False, use_aux=True)
    model.train()
    x = torch.randn(2, 3, 512, 512)
    main_out, aux_out = model(x)
    print("main_out:", main_out.shape)
    print("aux_out :", aux_out.shape)
