#!/usr/bin/env python

import argparse
from itertools import count
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import preprocessing

from tools import *

# Fix random seeds for reproducibility
torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SleepinessDataset(Dataset):
	'''
	This class defined the Sleepiness dataset. It
	- loads the features and the labels
	- converts them to torch tensors
	- normalizes the data (remove it in case you have done it already)
	'''

	def __init__(self, data_files, label_files):
		'''
		data_files and label_files should be in the same format as decribed
		in rn_baseline.py in run_svm function.
		'''

		# # Load data and labels
		X_train, X_dev, X_test = load_data(data_files)
		y_train, y_dev = load_labels(label_files)

		# # Data Pre-processing - Assuming data hasn't been pre-processed yet
		'''
		Data Pre-processing:
		- This is an important step when preparing data for a classifier.
		- Normalizing or transforming data can greatly help the classifier achieve better results and train faster.
		- Sklearn has a several preprocessing functions that can be used to this end:
		- https://scikit-learn.org/stable/modules/preprocessing.html

		- If you have not done the feature processing at Part 1 - now it is a good time to do it.
		'''

		self.X = torch.tensor(X_train, dtype=torch.float).to(device)
		self.y = torch.tensor(y_train, dtype=torch.long).to(device)

		self.dev_X = torch.tensor(X_dev, dtype=torch.float).to(device)
		self.dev_y = torch.tensor(y_dev, dtype=torch.long).to(device)

		self.test_X = torch.tensor(X_test, dtype=torch.float).to(device)


	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]



class FeedforwardNetwork(nn.Module):
	def __init__(
			self, n_classes, n_features, dropout, **kwargs):
		'''
		This function initializes the network. It defines its architecture.
			- n_classes (int): number of classes. In this problem it will be 2
			- n_features (int): number of features
			- dropout (float): dropout probability
		'''
		super(FeedforwardNetwork, self).__init__()

		'''
		The following block contains one linear layer and one activation function.

		One Linear layer is generically defined as nn.Linear(input_size, output_size).
		The number of neurons in the layer corresponds to the ouput size. Increasing the
		number of neurons in a layer increases the capability of the network to model the
		desired function. However, a very high number of neurons may lead the network to
		overfit, especially in situations where the training set is small.

		The activation functions add nonlinearities to the network. Some examples are:
		nn.ReLU(), nn.Tanh(), nn.Softmax().

		Between the nn.Linear() and the activation function, it is usual to include
		nn.BatchNorm1d(hidden_size), and after the adctivation function, it is usual to
		include nn.Dropout(p) to regularize the network.
		'''

		torch.manual_seed(1234)
		self.lin1 = nn.Sequential(
			nn.Linear(n_features, 32), #You may change the output size (2.3)
			# TODO: Introduce batch normalization (2.2)
			nn.ReLU()
			# TODO: Introduce dropout (2.2)
			)
		
		torch.manual_seed(1234)
		# self.lin2 = # TODO add one extra sequential with linear layer and activation function (2.1)

		torch.manual_seed(1234)
		# self.lin3 = # TODO add one extra sequential with linear layer and activation function (2.1)

		# for classification tasks you should use a softmax as final
		# activation layer, but if you use the loss function
		# nn.CrossEntropyLoss() as we are using in this lab, you do
		# not need to compute it explicitly
		torch.manual_seed(1234)
		self.lin_out = nn.Linear(32, n_classes)

	def forward(self, x, **kwargs):
		"""
		This function corresponds to the forward pass, which means
		that the input is being propagated through the network, layer
		by layer.
			- x (batch_size x n_features): a batch of training examples
		"""

		output = self.lin1(x)
		# output = self.lin2(output)
		# output = self.lin3(output)
		output = self.lin_out(output)

		return output



def train_batch(X, y, model, optimizer, criterion, **kwargs):
	"""
	X (n_examples x n_features)
	y (n_examples): gold labels
	model: a PyTorch defined model
	optimizer: optimizer used in gradient step
	criterion: loss function
	"""

	model.train()

	# zero the parameter gradients
	optimizer.zero_grad()

	# forward step
	outputs = ''' TODO '''

	loss = criterion(outputs, y)

	# propagate loss backward
	loss.backward()

	# updtate the weights
	optimizer.step()

	return loss


def predict(model, X):
	"""X (n_examples x n_features)"""
	model.eval()
	# make the predictions
	scores = ''' TODO '''

	# scores contains, for each example, two scores that can be interpreted as the
	# probability of each example belonging to each of the classes. To select the
	# final predicted label, we will select the class with higher probability.
	predicted_labels = scores.argmax(dim=-1)  # predicted_labels shape: (n_examples)

	return predicted_labels


def evaluate(model, X, y):
	"""
	X (n_examples x n_features)
	y (n_examples): labels
	"""
	model.eval()

	# make the predictions
	y_hat = ''' TODO '''

	# convert to cpu
	y_hat = y_hat.detach().cpu()
	y = y.detach().cpu()

	# compute evaluation metrics
	accuracy = accuracy_score(y, y_hat)
	prf 	 = precision_recall_fscore_support(y, y_hat, labels=[0,1], average='macro')

	return accuracy, prf


def train(dataset, model, optimizer, criterion, batch_size, epochs):

	train_dataloader = DataLoader(
		dataset, batch_size=batch_size, shuffle=True)

	dev_X, dev_y = dataset.dev_X, dataset.dev_y

	epochs = torch.arange(1, epochs + 1)
	train_mean_losses = []
	valid_accs = []
	train_losses = []
	valid_uar = []

	for ii in epochs:
		print('Training epoch {}'.format(ii))
		for X_batch, y_batch in train_dataloader:

			# train each batch:
			loss = '''TODO'''
			train_losses.append(loss)

		mean_loss = torch.tensor(train_losses).mean().item()
		print('Training loss: %.4f' % (mean_loss))

		train_mean_losses.append(mean_loss)

		# at the end of each epoch, evaluate with the dev set:
		val_accuracy, val_prf = '''TODO'''

		valid_accs.append(val_accuracy)
		valid_uar.append(val_prf[1])

		print('Valid acc: %.4f' % (valid_accs[-1]))
		print('Valid prf: ', val_prf)

	return model, train_mean_losses, valid_accs, valid_uar

