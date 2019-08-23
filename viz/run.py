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
						loss += criterion(output, target, reduction='sum').item() # sum up batch loss
						pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
						total_correct += pred.eq(target.view_as(pred)).sum().item()
		
		return {"accuracy": total_correct / len(loader.dataset), 
						"loss": loss / len(loader.dataset), 
						"total_correct": total_correct}


def run():
	# Massage the data
	normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
	train_dataset = datasets.MNIST('data', train=True, download=True, transform=normalize)
	test_dataset = datasets.MNIST('data', train=False, transform=normalize)
	# Configure data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
	first_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FIRST_EPOCH_BATCH_SIZE, shuffle=True)

	# For this example we will use the default values. 
	# See MNISTSparseCNN documentation for all possible parameters and their values.
	trained_model_file = "sparse_cnn_1ep_trained.pt"
	epoch_trained_model_file = "sparse_cnn_15ep_trained.pt"
	if os.path.exists(epoch_trained_model_file) and os.path.isfile(epoch_trained_model_file):
		print("Loading model from {}".format(epoch_trained_model_file))
		model = torch.load(epoch_trained_model_file)
		sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
		lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1, gamma=LEARNING_RATE_GAMMA)
		print(model)
	elif os.path.exists(trained_model_file) and os.path.isfile(trained_model_file):
		print("Loading model from {}".format(trained_model_file))
		model = torch.load(trained_model_file)
		sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
		lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1, gamma=LEARNING_RATE_GAMMA)
		print(model)
		test(model=model, loader=test_loader, criterion=F.nll_loss)
		for epoch in range(1, EPOCHS):
			train(model=model, loader=train_loader, optimizer=sgd, criterion=F.nll_loss)
			lr_scheduler.step()
			model.apply(rezero_weights)
			model.apply(update_boost_strength)
			results = test(model=model, loader=test_loader, criterion=F.nll_loss)
			print(results)
		torch.save(model, "sparse_cnn_{}ep_trained.pt".format(EPOCHS))
	else:
		print("Creating new model")
		model = MNISTSparseCNN().to(device)
		print(model)
		# TRAIN 1st Epoch
		sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
		lr_scheduler = optim.lr_scheduler.StepLR(sgd, step_size=1, gamma=LEARNING_RATE_GAMMA)
		train(model=model, loader=first_loader, optimizer=sgd, criterion=F.nll_loss)
		lr_scheduler.step()
		# Zero weights and apply boosting
		model.apply(rezero_weights)
		model.apply(update_boost_strength)
		# Save model for later
		torch.save(model, "sparse_cnn_1ep_trained.pt")

	


if __name__ == "__main__":
	run()

