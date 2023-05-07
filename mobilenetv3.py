import torchvision
import torch
import torch.nn as nn
import torchvision



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

        if self.training is True:
            return self.extractions
        return x

def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)