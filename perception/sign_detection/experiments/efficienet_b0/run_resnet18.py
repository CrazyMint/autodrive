import wandb
from wandb_utils import make, model_pipeline
from config import *

def run(config: dict):

    # login to wandb
    wandb.login()

    # launch experiment
    model = model_pipeline('traffic sign', True, config)

    return model

if __name__ == "__main__":

    model = run(config)