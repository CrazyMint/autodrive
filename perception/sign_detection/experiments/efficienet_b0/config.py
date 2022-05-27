# Build Configuration File
import torch
import torchvision.transforms as transforms

# Create Train Transformations
train_transform = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(10)]), p=0.5),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomAutocontrast(),
        transforms.RandomAffine(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]
)

# Create Test Transforms
test_transform = transforms.Compose(
    [
        #transforms.RandomRotation(180),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]
)

config = {
    ################# Training Hyperparameters ######################
    'num_epochs': 400,
    'batch_size': 128,
    'device': 'cuda',
    'learning_rate': 0.003,
    'checkpoint_file_name': 'checkpoint.pth',
    'load_checkpoint': False,
    'project': 'ich',
    'num_workers': 8,
    'seed': 42,
    'load_optimizer': False,
    'patience': 10,
    'pretrained': True,

    ################# Data Paths ######################
    'train_csv_path': '../../csv_files/train.csv', # this will be the larger training csv ,
    'test_csv_path': '../../csv_files/test.csv', # this will be the smaller testing csv

}
