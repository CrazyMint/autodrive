import torch.nn as nn
import torchvision


####################################################################################################


def build_model(config):

    model = torchvision.models.squeezenet1_0(pretrained=True)
    model.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
    model.classifier[1] = nn.Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))

    return model.to(config.device)


def build_test_model(config):

    model = torchvision.models.resnet18(pretrained=config['pretrained'])
    model.fc = nn.Linear(in_features=512, out_features=6, bias=True)

    return model.to(config['device'])

























