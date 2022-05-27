import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader


##########################################################################################################
class ICHDataset(Dataset):

    def __init__(self, data_csv, transforms) -> None:
        '''
        Args:
            :param data_path: (string) path where the image data is
            :param data_csv: (string) path to the csv file
            :param transforms: (callable, optional)
        Return:
            :return None
        '''

        self._data_csv = pd.read_csv(data_csv)
        self._data_csv.sample(frac=1).reset_index(drop=True)
        self._transforms = transforms

    def __len__(self):
        return len(self._data_csv)

    def __getitem__(self, index):

        data = self._data_csv.iloc[index]
        image_name = data['path']
        target = data['label']
        image = Image.open(image_name)

        if self._transforms:
            image = self._transforms(image)

        return image, torch.tensor(target)

##################################################################################################


def build_dataloader(config, dataset, shuffle, sampler=None):
    '''
    A simple function to return a PyTorch DataLoader object

    args:
        dataset -- a torch.utils.data.Dataset object
        batch_size [int] -- int specifying the batch size
        shuffle [bool] -- specifies if the data should be shuffled or not
        num_workers [int] -- specifies the number of workers to be used

    '''

    if sampler:
        return DataLoader(
            dataset = dataset,
            batch_size = config.batch_size,
            num_workers = config.num_workers,
            sampler=sampler,
        )

    else:
        return DataLoader(
            dataset = dataset,
            batch_size = config.batch_size,
            num_workers = config.num_workers,
            shuffle = shuffle,
        )


##################################################################################################
def build_test_dataloader(config, dataset):
    '''
    A simple function to return a PyTorch DataLoader object

    args:
        dataset -- a torch.utils.data.Dataset object
        batch_size [int] -- int specifying the batch size
        shuffle [bool] -- specifies if the data should be shuffled or not
        num_workers [int] -- specifies the number of workers to be used

    '''

    return DataLoader(
        dataset = dataset,
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        shuffle = False,
    )