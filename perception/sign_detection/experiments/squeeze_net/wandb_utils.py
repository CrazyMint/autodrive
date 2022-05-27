import os
import torch
import wandb
from train_functions import train_val
from losses import MultiTaskLoss
from config import train_transform, test_transform
from dataset import ICHDataset, build_dataloader
from model_architecture import build_model
from utils import load_checkpoint, seed_everything, initialize_weights
from timm import optim
##################################################################################################


def make(config):

    # set seed
    seed_everything(config.seed)

    # build training dataset & training data loader
    trainset = ICHDataset(config.train_csv_path, train_transform)
    trainloader = build_dataloader(config, trainset, True, None)

    # build validation dataset & validation data loader
    testset = ICHDataset(config.test_csv_path, test_transform)
    testloader = build_dataloader(config, testset, False, None)

    # build the Model
    model = build_model(config)
    #model.apply(initialize_weights)

    # print parameter count
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nmodel parameter count = ', pytorch_total_params)

    # set up the loss function
    criterion = MultiTaskLoss()

    # set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # load checkpoint
    if config.load_checkpoint:
        # check to see if file is available
        if os.path.isfile(config.checkpoint_file_name):
            model = build_model(config)
            model, optimizer = load_checkpoint(config,
                                               model,
                                               optimizer,
                                               config.load_optimizer)

    # send model to device
    model = model.to(config.device)

    # set up a scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.1,
                                                           patience=config.patience,
                                                           verbose=True,
                                                           min_lr=1e-6)
    # create a single dict to hold all parameters
    storage = {
        'model': model,
        'trainloader': trainloader,
        'testloader': testloader,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

    # return
    return storage

##################################################################################################


def model_pipeline(project_name, resume, config=None):

    # tell wandb to get started
    with wandb.init(project=project_name, config=config, resume=resume):

        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        print('[info]: creating model')
        storage = make(config)
        print('[info]: finished creating model \n')

        # and use them to train the model
        print('[info] starting training & validation')
        train_val(config,
                  storage['model'],
                  storage['optimizer'],
                  storage['criterion'],
                  storage['trainloader'],
                  storage['testloader'],
                  scheduler=storage['scheduler'],
                  save_acc=False)

        # and test its final performance
        #test(model, test_loader)

        wandb.save('test_sweep')

        #torch.onnx.export(model, "model.onnx")
        #wandb.save("model.onnx")

    return storage['model']













#%%
