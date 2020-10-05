import torch
from torch import nn
from torchvision import models as torch_models
import torch.nn.functional as F
from collections import OrderedDict

class BaselineNet(nn.Module):
    def __init__(self, in_features=1, num_classes=3, pre_train=False):
        super(BaselineNet, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.num_features = 128
        self.pre_train_num_classes = 2
        self.pre_train = pre_train
        self.fea_extractor = FeaExtractor2(self.in_features, self.num_features, self.pre_train_num_classes, self.pre_train)
        self.classifier = Classifier(self.num_features, self.num_classes)
    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.fea_extractor.load_state_dict(new_chkpt)

    def forward(self, x):
        x = self.fea_extractor(x)
        out = self.classifier(x)

        return out

def weight_init2(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


# feature extractor class
# this class can also be trained for self-supervision
class FeaExtractor2(nn.Module):
    def __init__(self, in_features=1, num_features=10, num_classes=None, pre_train=False):
        super(FeaExtractor2, self).__init__()

        self.in_features = in_features
        self.num_features = num_features
        self.num_classes = num_classes
        self.pre_train = pre_train
        in_c = self.in_features
        STEP = 5
        out_c = in_c + STEP
        for i in range(5):
            if i == 4:
                out_c = self.num_features
            self.add_module('conv{}'.format(i), nn.Conv2d(in_c, out_c, 3, 1, 1))
            if i != 4:
                in_c = out_c
                out_c = in_c + STEP
        self.conv5 = nn.Conv2d(out_c, self.num_classes, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(2*1*1, 2)

        weight_init2(self)
    def forward(self, x):
        x = self.pool(self.conv0(x))
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        if self.pre_train is False:
            x = self.adaptive_avg_pool(x)
            x = torch.flatten(x, 1)
            return x

        x = self.pool(self.conv5(x))
        x = self.avg_pool(x)
        
        return x


class Classifier(nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super(Classifier, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.num_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes)

    def forward(self, x, feat=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.mean(dim=0, keepdim=True)
        x = self.fc3(x)

        return x


class BaselineNet2(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(BaselineNet2, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.resnet = ResNet_extractor(layers, train_res4)
        self.classifier = Classifier(dim_dict[layers], num_classes)


    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.resnet.load_state_dict(new_chkpt)

    def forward(self, x):
        x = self.resnet(x)
        out = self.classifier(x)
        return out

class ResNet_extractor(nn.Module):
    def __init__(self, layers=50, train_res4=True):
        super().__init__()
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

        for param in self.resnet.parameters():
            param.requires_grad = False

        if train_res4:  # train res4
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.avgpool.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


