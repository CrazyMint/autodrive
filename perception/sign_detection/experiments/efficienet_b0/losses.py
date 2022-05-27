import torch.nn as nn


class MultiTaskLoss:

    def __init__(self):

        self._loss = nn.CrossEntropyLoss(reduction='mean')

    def  __call__(self, predictions, targets):

        # MAKE SURE WHEN YOU INSTANTIATE BCE LOSS, SET REDUCTION TO MEAN
        # ALPHA IS A DICT WITH VALUES ON HOW TO SCALE THE LOSSES
        # .. THIS CAN BE BASED ON THE OCCURENCES OF EACH CLASS * READ INTO THIS

        loss = self._loss(predictions, targets)

        return loss


