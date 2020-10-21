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
        self.num_features = 256
        self.pre_train = pre_train
        self.fea_extractor = FeaExtractorContextRestore2(self.in_features, self.num_features, self.pre_train)
        self.classifier = Classifier(self.num_features, self.num_classes)
        '''
        model1 = self.fea_extractor
        model2 = self.classifier
        cnt1 = cnt2 = 0
        for param in model1.parameters():
            cnt1 += param.numel()
            #cnt1 += 1
        for param in model2.parameters():
            cnt2 += param.numel()
            #cnt2 += 1
        print('total params in fea_ex2:', cnt1)
        print('total params in self.classifier:', cnt2)
        print('total params in BaselineNet:', (cnt1 + cnt2))
        cnt1_train =  sum(p.numel() for p in model1.parameters() if p.requires_grad)
        cnt2_train =  sum(p.numel() for p in model2.parameters() if p.requires_grad)
        print('the number of trainable  parameters in fea_ex2:', cnt1_train)
        print('the number of trainable parameters in self.classifier:', cnt2_train)
        print('the number of trainable parameters in BaselineNet:', (cnt1_train+cnt2_train))
        raise ValueError('Exit!!!')
        '''


        for param in self.fea_extractor.parameters():
            param.requires_grad = False

        for param in self.fea_extractor.conv10.parameters():
            param.requires_grad = True
        for param in self.fea_extractor.conv11.parameters():
            param.requires_grad = True

    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)['model']
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.fea_extractor.load_state_dict(new_chkpt)

    def forward(self, x):
        x = self.fea_extractor(x)
        out = self.classifier(x)

        return out



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

        self.resnet = ResNet_extractor2(layers, train_res4, 2, False)
        self.classifier = Classifier(dim_dict[layers], num_classes)
        model1 = self.resnet
        model2 = self.classifier
        cnt1 = cnt2 = 0
        for param in model1.parameters():
            cnt1 += param.numel()
        for param in model2.parameters():
            cnt2 += param.numel()
        print('total params in resnet:', cnt1)
        print('total params in self.classifier:', cnt2)
        print('the number of parameters in resnet:', cnt1 + cnt2)
        cnt1_train =  sum(p.numel() for p in model1.parameters() if p.requires_grad)
        cnt2_train =  sum(p.numel() for p in model2.parameters() if p.requires_grad)
        print('the number of trainable  parameters in resnet:', cnt1_train)
        print('the number of trainable parameters in self.classifier:', cnt2_train)
        print('the number of trainable parameters in BaselineNet2:', (cnt1_train+cnt2_train))


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

class ResNet_extractor2(nn.Module):
    def __init__(self, layers=18, train_res4=True, num_classes=2, pre_train=False):
        super().__init__()
        self.num_classes = num_classes
        self.pre_train = pre_train
        
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer2.parameters():
            param.requires_grad = True


        self.fc = nn.Linear(512, self.num_classes)
    
    def get_resnet(self):
        return self.resnet

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
        if self.pre_train is False:
            return x
        else:
            return self.fc(x)
        

class FeaExtractorContextRestore2(nn.Module):
    def __init__(self, in_features=1, num_features=10, pre_train=False):
        super(FeaExtractorContextRestore2, self).__init__()

        self.in_features = in_features
        self.num_features = num_features
        self.pre_train = pre_train
        
        self.conv1 = nn.Conv2d(self.in_features, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv10 = nn.Conv2d(64, self.num_features, 3, 1, 1)
        self.conv11 = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.up1 = nn.Conv2d(self.num_features, 64, 3, 1, 1)
        self.up2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.up3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.up4 = nn.Conv2d(16, self.in_features, 3, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):        

        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv5(self.conv4(x)))
        x = self.pool(self.conv11(self.conv10(x)))
        if self.pre_train is False:
            x = self.adaptive_avg_pool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            return self.up4(self.upsample(self.up3(self.upsample(self.up2(self.upsample(self.up1(x)))))))

