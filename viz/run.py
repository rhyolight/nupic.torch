import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm_notebook as tqdm

from nupic.torch.models import MNISTSparseCNN
from nupic.torch.modules import rezero_weights, update_boost_strength

# Training parameters
LEARNING_RATE = 0.02
LEARNING_RATE_GAMMA = 0.8
MOMENTUM = 0.0
EPOCHS = 15
FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000

SEED = 18
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer, criterion):
    """
    Train the model using given dataset loader. 
		Train the model using given dataset loader. 
    Train the model using given dataset loader. 
    Called on every epoch.
    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: dataloader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
    :type optimizer: :class:`torch.optim.Optimizer`
    :param criterion: loss function to use
    :type criterion: function
    """
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(loader, leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, loader, criterion):
    """
    Evaluate pre-trained model using given dataset loader.
    Called on every epoch.
    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: dataloader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param criterion: loss function to use
    :type criterion: function
    :return: Dict with "accuracy", "loss" and "total_correct"
    """
    model.eval()
    loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in tqdm(loader, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {
        "accuracy": total_correct / len(loader.dataset),
        "loss": loss / len(loader.dataset),
        "loss": loss / len(loader.dataset),
        "loss": loss / len(loader.dataset),
        "total_correct": total_correct,
    }


def create_model(learning_rate, learning_rate_gamma, momentum, step_size=1, saved=None):
    if saved is not None:
        model = torch.load(saved)
    else:
        model = MNISTSparseCNN().to(device)
    sgd = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.StepLR(
        sgd, step_size=step_size, gamma=learning_rate_gamma
    )
    return model, sgd, lr_scheduler


def save_model(model, state, epoch, class_name):
    filename = "{class_name}_{state}_epoch_{epoch}.pt".format(**locals())
    print("Saving model to {}".format(filename))
    torch.save(model, filename)


def train_epoch(model, epoch, scheduler, loader, optimizer, loss_fn):
    train(model=model, loader=loader, optimizer=optimizer, criterion=loss_fn)
    scheduler.step()
    class_name = "MNISTSparseCNN"
    save_model(model, "dense", epoch, class_name)
    model.apply(rezero_weights)
    save_model(model, "sparse", epoch, class_name)
    model.apply(update_boost_strength)
    save_model(model, "boosted", epoch, class_name)


def run():
    # Massage the data
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=normalize
    )
    test_dataset = datasets.MNIST("data", train=False, transform=normalize)

    # Configure data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True
    )
    first_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FIRST_EPOCH_BATCH_SIZE, shuffle=True
    )

    print("Creating new model...")
    (model, sgd, lr_scheduler) = create_model(
        LEARNING_RATE, LEARNING_RATE_GAMMA, MOMENTUM, step_size=1
    )

    print("Training 1st Epoch...")
    train_epoch(model, 0, lr_scheduler, first_loader, sgd, F.nll_loss)

    for epoch in range(1, EPOCHS):
        print("\tTraining epoch {}...".format(epoch))
        train_epoch(model, epoch, lr_scheduler, train_loader, sgd, F.nll_loss)
        results = test(model=model, loader=test_loader, criterion=F.nll_loss)
        print(results)


if __name__ == "__main__":
    run()
