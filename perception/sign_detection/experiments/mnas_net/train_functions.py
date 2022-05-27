import torch
from rich import print
from utils import random_translate, log_metrics, save_checkpoint
import wandb
from tqdm import tqdm

##################################################################################################


def train_step(config,
               model,
               optimizer,
               criterion,
               train_dataloader,
               **kwargs):

    # Initialize the training loss for the current Epoch
    epoch_loss = 0.0

    # Initialize variable to track number of training images
    data_count = 0

    # set model to train mode
    model.train()

    for index, (data) in enumerate(train_dataloader):

        # get data ex: (data, target)
        data, labels = data

        if kwargs['random_translate']:
            train_data = random_translate(data)

        # send data to device
        data = data.to(config.device)
        labels = labels.to(config.device)

        # zero out existing gradients
        optimizer.zero_grad()

        # forward pass
        predicted = model.forward(data)

        # calculate loss
        loss = criterion(predicted, labels)

        # backward pass
        loss.backward()

        # update gradients
        optimizer.step()

        # update loss
        epoch_loss += loss.item()

        # update number of training data seen
        data_count += 1

        del data

    return epoch_loss/data_count, data_count

##################################################################################################


def val_step(config,
             model,
             criterion,
             val_dataloader,
             **kwargs):

    # Initialize the training loss for the current Epoch
    epoch_loss = 0.0

    # Initialize variable to track number of training images
    data_count = 0

    # Initialize variable to keep track of number of correct outputs
    total_correct = 0

    # set model to train mode
    model.eval()

    with torch.no_grad():
        for index, (data) in enumerate(val_dataloader):

            # get data ex: (data, target)
            data, labels = data

            if kwargs['random_translate']:
                data = random_translate(data)

            # send data to device
            data = data.to(config.device)
            labels = labels.to(config.device)

            # forward pass
            predicted = model.forward(data)

            # calculate loss
            loss = criterion(predicted, labels)

            # run a test to get the accuracy if applicable
            if kwargs['run_test']:
                correct = val_test(predicted, labels)
                # keep track of number of total correct
                total_correct += correct

            # update loss for the current batch
            epoch_loss += loss.item()

            # update number of training data seen
            data_count += 1

        del data

        if kwargs['run_test']:
            return total_correct/data_count, epoch_loss/data_count,  data_count
        else:
            return None, epoch_loss/data_count, data_count,

##################################################################################################


def val_test(predicted, labels):
    pass

##################################################################################################


def train_val(config,
              model,
              optimizer,
              criterion,
              train_dataloader,
              val_dataloader,
              scheduler=None,
              **kwargs):

    # Tell wandb to watch the models and optimizer values
    wandb.watch(
        model,
        criterion,
        log='all',
        log_freq=10,
        log_graph=True
    )

    # Initialize Best validation Loss and Accuracy
    best_val_loss = 100.0
    best_val_acc = 0.0
    accuracy = 0

    # Run Training and Validation
    for epoch in tqdm(range(config.num_epochs)):

        save = False

        # Run a single Training Step
        train_loss, train_data_count = train_step(
            config,
            model,
            optimizer,
            criterion,
            train_dataloader,
            random_translate=None)

        # Run a Single Validation Step
        _, val_loss, val_data_count = val_step(
            config,
            model,
            criterion,
            val_dataloader,
            random_translate=None,
            run_test=False)

        # log metrics back to wandb
        log_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_example_count=train_data_count)

        # save/display information
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save = True

        if accuracy > best_val_acc:
            best_val_acc = accuracy


        if kwargs['save_acc']:
            save_checkpoint(model, optimizer, 'checkpoint.pth')
            print(f"=> epoch -- {epoch} || training loss -- {train_loss} || validation loss -- {val_loss} || validation acc -- {accuracy}")

            # step scheduler
            scheduler.step(accuracy)
        else:
            if save:
                save_checkpoint(model, optimizer, 'best_loss_model.pth')
                print(f"=> epoch -- {epoch} || training loss -- {train_loss} || validation loss -- {val_loss} -- saved")
            else:
                print(f"=> epoch -- {epoch} || training loss -- {train_loss} || validation loss -- {val_loss}")

            scheduler.step(val_loss)

































#%%
