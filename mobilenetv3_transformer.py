import torchvision
import torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from transformer_cosine import TransformerEncoder, TransformerEncoderLayer


class CustomMobileNetV3(nn.Module):
    def __init__(self):
        super(CustomMobileNetV3, self).__init__()
        original_model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        self.layers0 = original_model.features[0] # 0
        self.transform0 = feature_transform(16, 64)
        self.layers1 = original_model.features[1] # 1
        self.transform1 = feature_transform(16, 128)
        self.layers2 = original_model.features[2:4] # 2-3
        self.transform2 = feature_transform(24, 256)
        self.layers3 = original_model.features[4:8] # 4-7
        self.transform3 = feature_transform(48, 512)
        self.layers4 = original_model.features[8:12] # 8-11
        self.transform4 = feature_transform(96, 512)
        self.layers5 = original_model.features[12] # 12
        self.layers6 = original_model.avgpool # 12
        self.layers7 = nn.Conv2d(576, 512, kernel_size=3, stride=1, padding=1)


        d_model = 512
        nhead = 2
        num_layers = 1
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        self.extractions = []
        x = self.layers0(x)
        self.extractions.append(self.transform0(x))
        x = self.layers1(x)
        self.extractions.append(self.transform1(x))
        x = self.layers2(x)
        self.extractions.append(self.transform2(x))
        x = self.layers3(x)
        self.extractions.append(self.transform3(x))
        x = self.layers4(x)
        self.extractions.append(self.transform4(x))
        x = self.layers5(x)
        x = self.layers6(x)
        x = self.layers7(x)

        if self.training is True:
            return self.extractions

        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, features = self.encoder(x, (h,w))   # transformer
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        #
        x = F.upsample_bilinear(x, size=(rh, rw))
        x = self.reg_layer_0(x)   # regression head
        return torch.relu(x), features


def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)

