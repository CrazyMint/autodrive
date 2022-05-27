from model_architecture import build_test_model
from config import *
from dataset import ICHDataset, build_test_dataloader
from tqdm import tqdm
from utils import random_translate
from sklearn.metrics import accuracy_score
from rich import print

##################################################################################################

def get_acc(pred, target, name):

    y_true = target[name].detach().cpu().numpy()
    y_pred = pred[name].sigmoid().round().detach().cpu().numpy()

    return accuracy_score(y_true, y_pred,)


##################################################################################################

def test(model, testloader, **kwargs):

    total = 0

    # category wise accuracy
    epidural_acc = 0.0
    intraparenchymal_acc = 0.0
    intraventricular_acc = 0.0
    subarachnoid_acc = 0.0
    subdural_acc = 0.0
    any_acc = 0.0

    model.eval()

    with torch.no_grad():
        for idx, (data) in tqdm(enumerate(testloader)):

            # get data ex: (data, target)
            data, labels = data

            if kwargs['random_translate']:
                data = random_translate(data)

            data = data.to(config['device'])
            for key, value in labels.items():
                labels[key] = labels[key].float().to(config['device'])

            # forward pass
            predicted = model.forward(data)

            # update accuracies
            epidural_acc += get_acc(predicted, labels, 'epidural')
            intraparenchymal_acc += get_acc(predicted, labels, 'intraparenchymal')
            intraventricular_acc += get_acc(predicted, labels, 'intraventricular')
            subarachnoid_acc += get_acc(predicted, labels, 'subarachnoid')
            subdural_acc += get_acc(predicted, labels, 'subdural')
            any_acc += get_acc(predicted, labels, 'any')

            total += 1

    print(f"epidural accuracy: {epidural_acc / total}")
    print(f"intraparenchymal accuracy: {intraparenchymal_acc / total}")
    print(f"intraventricular accuracy: {intraventricular_acc / total}")
    print(f"subarachnoid accuracy: {subarachnoid_acc / total}")
    print(f"subdural accuracy: {subdural_acc / total}")
    print(f"any accuracy: {any_acc / total}")

def run(config):

    # Build Dataset & Dataloader
    testset = ICHDataset(config['test_data_path'], config['test_csv_path'],  test_transform)
    testloader = build_test_dataloader(config, testset)
    print('test data set size: ', testset.__len__())

    # Build Model
    model = build_test_model(config)
    checkpoint = torch.load('best_loss_model9112.pth', map_location='cuda')
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to('cuda')

    test(model, testloader, random_translate=None)


if __name__ == "__main__":

    model = run(config)

