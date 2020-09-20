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
        #self.pre_train_num_classes = 4
        self.pre_train_num_classes = 2
        self.pre_train = pre_train
        #self.fea_extractor = FeaExtractor(self.in_features, self.num_features, self.num_classes, self.pre_train)
        self.fea_extractor = FeaExtractor2(self.in_features, self.num_features, self.pre_train_num_classes, self.pre_train)
        #self.fea_extractor = FeaExtractorContextRestore(self.in_features, self.num_features, self.pre_train)
        #self.fea_extractor = FeaExtractorContextRestore2(self.in_features, self.num_features, self.pre_train)
        self.classifier = Classifier(self.num_features, self.num_classes)
    '''
    def set_fea_extractor(self, chkpt_path):
        chkpt = torch.load(chkpt_path)
        self.fea_extractor.load_state_dict(chkpt)
    '''

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


def weight_init(net):
    for name, m in net.named_children():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

def weight_init2(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

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
        #out_c = out_c - 30
        self.conv5 = nn.Conv2d(out_c, self.num_classes, 3, 1, 1)
        #self.conv5 = nn.Conv2d(out_c, self.num_classes, 4, 2, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)

        #weight_init(self)
        weight_init2(self)
    
    def forward(self, x):
        x = self.pool(self.conv0(x))
        #print('after conv0 and pool',x.size())
        x = self.pool(self.conv1(x))
        #print('after conv1 and pool',x.size())
        x = self.pool(self.conv2(x))
        #print('after conv2 and pool',x.size())
        x = self.pool(self.conv3(x))
        #print('after conv3 and pool',x.size())
        x = self.pool(self.conv4(x))
        #print('after conv4 and pool',x.size())
        # the if considition is met in the final prediction
        if self.pre_train is False:
            x = self.adaptive_avg_pool(x)
            #print('after avg_pool:',x.size())
            x = torch.flatten(x, 1)
            print('after flatten: ',x.size())
            return x

        x = self.pool(self.conv5(x))
        #print('after conv5 and pool',x.size())
        #x = self.pool(self.conv6(x))
        #x = self.conv8(x)
        #x = self.pool(x)
        x = self.avg_pool(x)
        #print(x.size())
        #x = self.softmax(x)
        #print(x.size())
        
        return x

class FeaExtractor3(nn.Module):
    def __init__(self, in_features=1, num_features=10, num_classes=None, pre_train=False):
        super(FeaExtractor3, self).__init__()

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
        #out_c = out_c - 30
        self.conv5 = nn.Conv2d(out_c, self.num_classes, 3, 1, 1)
        #self.conv5 = nn.Conv2d(out_c, self.num_classes, 4, 2, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)

        #weight_init(self)
        weight_init2(self)
    
    def forward(self, x):
        x = self.pool(self.conv0(x))
        #print('after conv0 and pool',x.size())
        x = self.pool(self.conv1(x))
        #print('after conv1 and pool',x.size())
        x = self.pool(self.conv2(x))
        #print('after conv2 and pool',x.size())
        x = self.pool(self.conv3(x))
        #print('after conv3 and pool',x.size())
        x = self.pool(self.conv4(x))
        #print('after conv4 and pool',x.size())
        # the if considition is met in the final prediction
        if self.pre_train is False:
            x = self.adaptive_avg_pool(x)
            #print('after avg_pool:',x.size())
            x = torch.flatten(x, 1)
            print('after flatten: ',x.size())
            return x

        x = self.pool(self.conv5(x))
        #print('after conv5 and pool',x.size())
        #x = self.pool(self.conv6(x))
        #x = self.conv8(x)
        #x = self.pool(x)
        x = self.avg_pool(x)
        #print(x.size())
        #x = self.softmax(x)
        #print(x.size())
        
        return x

# context
class FeaExtractorContextRestore(nn.Module):
    def __init__(self, in_features=1, num_features=10, pre_train=False):
        super(FeaExtractorContextRestore, self).__init__()

        self.in_features = in_features
        self.num_features = num_features
        self.pre_train = pre_train
        
        self.conv1 = nn.Conv2d(self.in_features, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv9 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv10 = nn.Conv2d(64, self.num_features, 3, 1, 1)
        self.conv11 = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
        self.conv12 = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.up1 = nn.ConvTranspose2d(self.num_features, 64, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(16, self.in_features, 2, stride=2)
        weight_init2(self)
    def forward(self, x):
        x = self.pool(self.conv3(self.conv2(self.conv1(x))))
        x = self.pool(self.conv6(self.conv5(self.conv4(x))))
        x = self.pool(self.conv9(self.conv8(self.conv7(x))))
        x = self.pool(self.conv12(self.conv11(self.conv10(x))))

        if self.pre_train is False:
            x = self.adaptive_avg_pool(x)
            x = torch.flatten(x, 1)
            #print('after flatten: ',x.size())
            return x
        else:
            return self.up4(self.up3(self.up2(self.up1(x))))

# feature extractor class
# this class can also be trained for self-supervision
class FeaExtractor(nn.Module):
    def __init__(self, in_features=1, num_features=10, num_classes=None, pre_train=False):
        super(FeaExtractor, self).__init__()

        self.in_features = in_features
        self.num_features = num_features
        self.num_classes = num_classes
        self.pre_train = pre_train
        self.conv1 = nn.Conv2d(self.in_features, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 53 * 53, 84)
        self.fc1 = nn.Linear(16 * 53 * 53, 128)
        self.fc2 = nn.Linear(128, self.num_features)
        self.fc3 = nn.Linear(self.num_features, self.num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x = torch.flatten(x2, 1)
        x3 = F.relu(self.fc3(x2))
        if self.pre_train is False:
            return x
        else:
            return x3

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

        #self.resnet = ResNet_extractor(layers, train_res4)
        self.resnet = ResNet_extractor2(layers, train_res4)
        #print(self.resnet)
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
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
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

        #print('x size before avg_pool: ', x.size())
        x = self.resnet.avgpool(x)
        #print('x size: ', x.size())
        x = torch.flatten(x, 1)
        #print('x size: ', x.size())
        return x

class ResNet_extractor2(nn.Module):
    def __init__(self, layers=18, train_res4=True, num_classes=2, pre_train=False):
        super().__init__()
        self.num_classes = num_classes
        self.pre_train = pre_train
        
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=False)
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

        for param in self.resnet.parameters():
            #param.requires_grad = True
            param.requires_grad = False

        if train_res4:  # train res4
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.avgpool.parameters():
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

        #print('x size before avg_pool: ', x.size())
        x = self.resnet.avgpool(x)
        #print('x size: ', x.size())

        x = torch.flatten(x, 1)
        if self.pre_train is False:
            #print('x size: ', x.size())
            return x
        else:
            return self.fc(x)
        

class FeaExtractorContextRestore2(nn.Module):
    def __init__(self, in_features=1, num_features=10, pre_train=False):
        super(FeaExtractorContextRestore2, self).__init__()

        self.in_features = in_features
        self.num_features = num_features
        self.pre_train = pre_train
        
        self.conv1 = nn.Conv2d(self.in_features, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        #self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        #self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        #self.conv7 = nn.Conv2d(32, 64, 3, 1, 1)
        #self.conv8 = nn.Conv2d(64, 64, 3, 1, 1)
        #self.conv9 = nn.Conv2d(64, 64, 3, 1, 1)
        #self.conv10 = nn.Conv2d(64, self.num_features, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, self.num_features, 3, 1, 1)
        self.conv11 = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
        #self.conv12 = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
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

        x = self.lrelu(self.conv2(self.conv1(x))) * 0.2 + x
        #x = self.lrelu(self.conv2(self.conv1(x)))
        x = self.lrelu(self.conv5(self.conv4(x))) 
        #x = self.lrelu(self.conv8(self.conv7(x)))
        x = self.lrelu(self.conv11(self.conv10(x)))
        '''
        
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv5(self.conv4(x)))
        #x = self.lrelu(self.conv8(self.conv7(x)))
        x = self.pool(self.conv11(self.conv10(x)))
        '''
        '''
        x = self.lrelu(self.conv3(self.conv2(self.conv1(x))))
        x = self.lrelu(self.conv6(self.conv5(self.conv4(x))))
        x = self.lrelu(self.conv9(self.conv8(self.conv7(x))))
        x = self.lrelu(self.conv12(self.conv11(self.conv10(x))))
        '''
        if self.pre_train is False:
            x = self.adaptive_avg_pool(x)
            x = torch.flatten(x, 1)
            #print('after flatten: ',x.size())
            return x
        else:
            #return self.up4(self.upsample(self.up3(self.upsample(self.up2(self.upsample(self.up1(x)))))))
            return self.up4(self.up3(self.up2(self.up1(x))))
            return self.fc(x)
