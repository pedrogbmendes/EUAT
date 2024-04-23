
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys

import torchvision.models as models_



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    


###
# code taken from https://github.com/landskape-ai/ImageNet-Downsampled
###
class BasicBlock(nn.Module):
    # code taken from https://github.com/landskape-ai/ImageNet-Downsampled
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # code taken from https://github.com/landskape-ai/ImageNet-Downsampled
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # code taken from https://github.com/landskape-ai/ImageNet-Downsampled
    def __init__(self, block, layers, num_classes, depth, dropout_rate=0.0):
        self.inplanes = 64
        self.dropout_rate = dropout_rate
        self.depth = depth
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        #if self.dropout_rat > 0:
        if depth == 18:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #if self.dropout_rate > 0: x = self.maxpool(x)

        x = self.layer1(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = self.layer2(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = self.layer3(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = self.layer4(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        self.dropout.train()
        #for m in model.modules():
        #    if m.__class__.__name__.startswith('Dropout'):
        #        m.train()





###
# code taken from https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/preact_resnet.py
###

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.0):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.dropout_rate > 0: x = self.maxpool(x)

        x = self.layer1(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = self.layer2(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = self.layer3(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = self.layer4(x)
        if self.dropout_rate > 0: x = self.dropout(x)

        x = F.relu(self.bn(x))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


    



###
# code taken from https://github.com/xternalz/WideResNet-pytorch
###
class BasicBlock1(nn.Module):
    # code taken from https://github.com/xternalz/WideResNet-pytorch
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    # code taken from https://github.com/xternalz/WideResNet-pytorch
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    # code taken from https://github.com/xternalz/WideResNet-pytorch
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock1
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)     


###
# code taken from https://github.com/tml-epfl/understanding-fast-adv-training
###
class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs


class PreActBlock1(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1):
        super(PreActBlock1, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def relu_with_stats(self, preact):
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())
        act = F.relu(preact)
        return act

    def forward(self, x):
        out = self.relu_with_stats(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu_with_stats(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet1(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False):
        super(PreActResNet1, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = 64
        self.avg_preact = None
        # self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1).cuda()
        # self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1).cuda()
        self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
        self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, n_cls)

        layers = [self.normalize, self.conv1, self.layer1[0].bn1]
        self.model_preact_hl1 = nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def calc_distances_hl1(self, X):
        bn1 = self.layer1[0].bn1
        weight_full = self.conv1.weight * bn1.weight.view(-1, 1, 1, 1) / (self.std * (bn1.running_var.view(-1, 1, 1, 1) + bn1.eps)**0.5)
        first_conv_norm_channelwise = weight_full.abs().sum((1, 2, 3))  # note: l1 distance is implemented!
        first_conv_norm_channelwise[first_conv_norm_channelwise < 1e-6] = np.nan
        distances = self.model_preact_hl1(X).abs() / first_conv_norm_channelwise[None, :, None, None]
        distances = distances.view(X.shape[0], -1)
        # # Sanity check
        # X.requires_grad = True
        # preact = self.model_preact_hl1(X)[:, 0, 10, 10].sum()  # for a unit sufficiently far from the boundary
        # grad = torch.autograd.grad(preact, X)[0]
        # grad_norm = grad.view(X.shape[0], -1).abs().sum(1)
        # print(grad_norm)
        # assert (first_conv_norm_channelwise[0] - grad_norm[0]).abs().item() < 1e-6
        return distances

    def forward(self, x):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        x = self.normalize(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        avg_preacts_all = []
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            avg_preacts_all += layer.avg_preacts
        self.avg_preact = np.mean(avg_preacts_all)

        return out



###
# code taken from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
###
def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        self.dropout.train()


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        #print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.dropout_rate = dropout_rate


    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for _layer in self.layer1:
            _layer.enable_dropout()
        for _layer in self.layer2:
            _layer.enable_dropout()
        for _layer in self.layer3:
            _layer.enable_dropout()




#class Resnet50(models_.resnet50()):
#    def enable_dropout(self):
#        return



# Define a custom ResNet-50 model with dropout
class ResNet50WithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(ResNet50WithDropout, self).__init__()
        # Load the pre-trained ResNet-50 model
        resnet = models_.resnet50(pretrained=False)
        
        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add dropout layers
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Add a new fully connected layer for the specified number of classes
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        self.dropout.train()



    
class CustomResNet50(models_.resnet.ResNet):
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        return
#        self.dropout.train()

