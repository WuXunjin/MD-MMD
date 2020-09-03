#coding=utf-8

from torch import nn, autograd
from torchvision import models
from munch import Munch
import torch
from loss import EntropyLoss, ClusterLoss, ClusterLoss1

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

class AlexNetExtractor2(nn.Module):
    def __init__(self):
        super(AlexNetExtractor2, self).__init__()
        model = models.alexnet(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool

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
                nn.ReLU(True)
                )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def out_features(self):
        return 4096

    def get_param_groups(self, learning_rate):
        #param_groups = []
        param_groups = [
            {"params": self.features.parameters(), "lr": learning_rate},
            {"params": self.avgpool.parameters(), "lr": learning_rate}]

        for i in range(6):
            param_groups += [{"params": self.classifier[i].parameters(), "lr": learning_rate * 10}]
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
        param_groups = [{"params": self.layer4.parameters(),  "lr": learning_rate}]
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
        param_groups = [{"params": self.layer4.parameters(),  "lr": learning_rate}]
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
        param_groups = [{"params": self.layer4.parameters(),  "lr": learning_rate}]
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
        param_groups = [{"params": self.layer4.parameters(),  "lr": learning_rate}]
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
        param_groups = [{"params": self.layer4.parameters(),  "lr": learning_rate}]
        return param_groups


class InceptionExtractor(nn.Module):
    def __init__(self):
        super(InceptionExtractor, self).__init__()
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        self.features = model
        self.features.fc = nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.shape[0], -1)
        return x

    def out_features(self):
        return 2048

    def get_param_groups(self, learning_rate=1e-3):
        return []

extractor_dict = Munch({
    "AlexNetExtractor":   AlexNetExtractor,
    "AlexNetExtractor2":  AlexNetExtractor2,
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

class NewAlexNetClassifier(nn.Module):
    def __init__(self, in_features=2048, num_class=2):
        super(NewAlexNetClassifier, self).__init__()
        self.num_class =num_class

        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, self.num_class)
                )

    def forward(self, x):
        return self.classifier(x)

    def get_param_groups(self, new_layer_learning_rate):
        param_groups = []
        for i in range(7):
            param_groups += [{"params": self.classifier[i].parameters(), "lr": new_layer_learning_rate}]
        return param_groups


classifier_dict = Munch({
    "SingleClassifier": SingleClassifier,
    "AlexNetClassifier": AlexNetClassifier,
    "NewAlexNetClassifier": NewAlexNetClassifier,
    })


# Multi Domain Cluster Layer -----------------------------------------------------------------
class MDCL(nn.Module):
    def __init__(self, in_features, out_features, latent_domain_num=2):
        super(MDCL, self).__init__()
        self.latent_domain_num = latent_domain_num
        self.in_features, self.out_features = in_features, out_features
        self.aux_classifier = nn.Linear(self.in_features, self.latent_domain_num)
        self.layers = []
        for i in range(self.latent_domain_num):
            self.layers.append(nn.Linear(self.in_features, self.out_features).to(DEVICE))
        self.cluster_ciriterion = ClusterLoss1()
        self.entropy_ciriterion = EntropyLoss()
        self.moving_center = None
        self.moving_factor = 0.9

    def forward(self, x):
        #combine loss
        batch_size = int(x.size()[0])
        features_size = self.out_features
        latent_domain = self.aux_classifier(x)
        aux_entropy_loss = self.entropy_ciriterion(latent_domain)
        latent_domain_label = nn.functional.softmax(latent_domain, dim=1)
        outputs = [layer(x) for layer in self.layers]
        stack_outputs = torch.stack(outputs, dim=2)
        now_center = (stack_outputs * latent_domain_label.unsqueeze(1).expand(
            batch_size, features_size, self.latent_domain_num))
        if torch.is_grad_enabled():
            self.moving_center = now_center if self.moving_center is None \
                                            else self.moving_center * self.moving_center + now_center * (1.0 - self.moving_factor)
        cluster_loss = self.cluster_ciriterion(stack_outputs, latent_domain_label)

        expand_latent_domain_label = latent_domain_label.unsqueeze(1).expand(batch_size, features_size, self.latent_domain_num)
        combine_outputs = (expand_latent_domain_label * stack_outputs).sum(2)
        return combine_outputs, cluster_loss, aux_entropy_loss

    def get_param_groups(self, learning_rate):
        param_groups = [{"params": self.aux_classifier.parameters(), "lr": learning_rate}]
        for layer in self.layers:
            param_groups.append({"params": layer.parameters(), "lr": learning_rate})
        return param_groups

def build_model(cfg):
    DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
    features = extractor_dict[cfg.extractor]().to(DEVICE)
    bottleneck_layer = MDCL(in_features=features.out_features(),
            out_features=cfg.bottleneck_size,
            latent_domain_num=cfg.latent_domain_num).to(DEVICE)
    classifier = classifier_dict[cfg.classifier](in_features=cfg.bottleneck_size,
            num_class=cfg.num_class).to(DEVICE)
    return features, bottleneck_layer, classifier

