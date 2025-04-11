import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, VGG16_BN_Weights, DenseNet121_Weights, Inception_V3_Weights, Swin_T_Weights

class LeNet5(nn.Module):
    def __init__(self, nb_channel=3):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(nb_channel, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self, is_pretrained=False):
        super(ResNet18, self).__init__()
        if is_pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18()

        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class VGGNet16(nn.Module):
    def __init__(self, is_pretrained=False):
        super(VGGNet16, self).__init__()
        if is_pretrained:
            self.vgg16 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).features
        else:
            self.vgg16 = models.vgg16_bn().features

    def forward(self, x):
        x = self.vgg16(x)
        return x
    
class DenseNet121(nn.Module):
    def __init__(self, is_pretrained=False):
        super(DenseNet121, self).__init__()
        if is_pretrained:
            self.densenet121 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features
        else:
            self.densenet121 = models.densenet121().features

    def forward(self, x):
        x = self.densenet121(x)
        return x    
    
class InceptionV3(nn.Module):
    def __init__(self, is_pretrained=False):
        super(InceptionV3, self).__init__()
        if is_pretrained:
            inception_v3 = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        else:
            inception_v3 = models.inception_v3()

        self.inception_v3 = nn.Sequential(*list(inception_v3.children())[:-2])

    def forward(self, x):
        x = self.inception_v3(x)
        return x    
    
class SwinTransformer(nn.Module):
    def __init__(self, is_pretrained=False):
        super(SwinTransformer, self).__init__()
        if is_pretrained:
            self.swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).features
        else:
            self.swin = models.swin_t().features

    def forward(self, x):
        outputs = self.swin(x)

        return outputs
    
class CNN_Model(nn.Module):
    def __init__(self,
                 feature_model='LeNet5',
                 drop_out=0.2,
                 input_size=(3, 64, 64),
                 hidden_size=1024,
                 layer_depth=3,
                 nb_classes=10,
                 is_pretrained=False,
                 ):
        super(CNN_Model, self).__init__()

        self.feature_model = feature_model
        self.input_size = input_size

        if feature_model == 'LeNet5':
            self.feature_extractor = LeNet5()
        elif feature_model == 'ResNet18':
            self.feature_extractor = ResNet18(is_pretrained)
        elif feature_model == 'VGGNet16':
            self.feature_extractor = VGGNet16(is_pretrained)
        elif feature_model == 'DenseNet121':
            self.feature_extractor = DenseNet121(is_pretrained)
        elif feature_model == 'InceptionV3':
            self.feature_extractor = InceptionV3(is_pretrained)
        elif feature_model == 'SwinTransformer':
            self.feature_extractor = SwinTransformer(is_pretrained)
            
        if feature_model != 'LeNet5' and is_pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.flatten = nn.Flatten()

        self.feature_dim = self._get_feature_dim()
        input_dim = self.feature_dim
        classifier_layers = []
        for i in range(layer_depth):
            classifier_layers.append(nn.Linear(input_dim, hidden_size))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(p=drop_out))
            input_dim = hidden_size
            
            hidden_size /= 2

            if hidden_size < 64:
                hidden_size *= 2

            hidden_size = int(hidden_size)

        classifier_layers.append(nn.Linear(input_dim, nb_classes))

        self.classifier = nn.Sequential(*classifier_layers)
        self.classifier.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def _get_feature_dim(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_size[0], self.input_size[1], self.input_size[2])
            features = self.feature_extractor(dummy_input)
            return features.view(1, -1).size(1)