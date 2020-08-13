#zzcoding=utf-8

from torch import nn, autograd
from torchvision import models
from munch import Munch
import torch


class SilenceLayer(autograd.Function):
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 * x

    def backward(self, gradOutput):
        return 0.0 * gradOutput

# Conv Layer Extractor -------------------------------------------------------------

class AlexNetExtractor(nn.Module):
    def __init__(self):
        super(AlexNetExtractor, self).__init__()

        model = models.alexnet(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 9216

    def get_param_groups(self, learning_rate):
        param_groups = [
                {"params": self.features.parameters(), "lr": learning_rate},
                {"params": self.avgpool.parameters(), "lr": learning_rate}]
        return param_groups

class ResNet18Extractor(nn.Module):
    def __init__(self):
        super(ResNet18Extractor, self).__init__()
        model = models.resnet18(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 512

    def get_param_groups(self, learning_rate):
        param_groups = [
                {"params": self.layer3.parameters(),  "lr": learning_rate},
                {"params": self.layer4.parameters(),  "lr": learning_rate},
                {"params": self.avgpool.parameters(), "lr": learning_rate}]
        return param_groups


class ResNet34Extractor(nn.Module):
    def __init__(self):
        super(ResNet34Extractor, self).__init__()
        model = models.resnet34(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 512

    def get_param_groups(self, learning_rate):
        param_groups = [
                {"params": self.layer3.parameters(),  "lr": learning_rate},
                {"params": self.layer4.parameters(),  "lr": learning_rate},
                {"params": self.avgpool.parameters(), "lr": learning_rate}]
        return param_groups


class ResNet50Extractor(nn.Module):
    def __init__(self):
        super(ResNet50Extractor, self).__init__()
        model = models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 2048

    def get_param_groups(self, learning_rate):
        param_groups = [
                {"params": self.layer3.parameters(),  "lr": learning_rate},
                {"params": self.layer4.parameters(),  "lr": learning_rate},
                {"params": self.avgpool.parameters(), "lr": learning_rate}]
        return param_groups


class ResNet101Extractor(nn.Module):
    def __init__(self):
        super(ResNet101Extractor, self).__init__()
        model = models.resnet101(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 2048

    def get_param_groups(self, learning_rate):
        param_groups = [
                {"params": self.layer3.parameters(),  "lr": learning_rate},
                {"params": self.layer4.parameters(),  "lr": learning_rate},
                {"params": self.avgpool.parameters(), "lr": learning_rate}]
        return param_groups


class ResNet152Extractor(nn.Module):
    def __init__(self):
        super(ResNet152Extractor, self).__init__()
        model = models.resnet152(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 2048

    def get_param_groups(self, learning_rate):
        param_groups = [
                {"params": self.layer3.parameters(),  "lr": learning_rate},
                {"params": self.layer4.parameters(),  "lr": learning_rate},
                {"params": self.avgpool.parameters(), "lr": learning_rate}]
        return param_groups


class InceptionExtractor(nn.Module):
    def __init__(self):
        super(InceptionExtractor, self).__init__()
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        self.features = model
        self.features.fc = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 2048

    def get_param_groups(self, learning_rate=1e-3):
        return []

extractor_dict = Munch({
    "AlexNetExtractor":   AlexNetExtractor,
    "ResNet18Extractor":  ResNet18Extractor,
    "ResNet34Extractor":  ResNet34Extractor,
    "ResNet50Extractor":  ResNet50Extractor,
    "ResNet101Extractor": ResNet101Extractor,
    "ResNet152Extractor": ResNet152Extractor,
    "InceptionExtractor": InceptionExtractor
    })

# Classifier layer -----------------------------------------------------------------

class SingleClassifier(nn.Module): # 简单的一层FC
    def __init__(self, in_features, num_class):
        super(SingleClassifier, self).__init__()
        self.num_class = num_class
        self.classifier = nn.Linear(in_features, self.num_class)

    def forward(self, x):
        x = self.classifier(x)
        return x

    def get_param_groups(self, learning_rate):
        param_groups = [{"params": self.classifier.parameters(), "lr": learning_rate}]
        return param_groups


class AlexNetClassifier(nn.Module):
    def __init__(self, in_features=9216, num_class=2):
        super(AlexNetClassifier, self).__init__()
        self.num_class =num_class
        model = models.alexnet(pretrained=True)

        fc1 = nn.Linear(9216, 4096)
        fc1.bias = model.classifier[1].bias
        fc1.weight = model.classifier[1].weight

        fc2 = nn.Linear(4096, 4096)
        fc2.bias = model.classifier[4].bias
        fc2.weight = model.classifier[4].weight

        self.classifier = nn.Sequential(
                nn.Dropout(),
                fc1,
                nn.ReLU(True),
                nn.Dropout(),
                fc2,
                nn.ReLU(True),
                nn.Linear(4096, self.num_class)
                )

    def forward(self, x):
        return self.classifier(x)

    def get_param_groups(self, learning_rate, new_layer_learning_rate):
        param_groups = []
        for i in range(6):
            param_groups += [{"params": self.classifier[i].parameters(), "lr": learning_rate}]
        param_groups += [{"params": self.classifier[-1].parameters(), "lr": new_layer_learning_rate}]
        return param_groups


classifier_dict = Munch({
    "SingleClassifier": SingleClassifier,
    "AlexNetClassifier": AlexNetClassifier
    })
