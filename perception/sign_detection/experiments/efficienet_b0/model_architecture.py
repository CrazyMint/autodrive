import torch.nn as nn
import torchvision


####################################################################################################


def build_model(config):

    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=18, bias=True)

    return model.to(config.device)


def build_test_model(config):

    model = torchvision.models.resnet18(pretrained=config['pretrained'])
    model.fc = nn.Linear(in_features=512, out_features=6, bias=True)

    return model.to(config['device'])

























