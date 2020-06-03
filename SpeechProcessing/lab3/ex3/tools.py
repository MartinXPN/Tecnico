import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Load labels from CSV file
def load_labels(files, datasets):
	frames = [pd.read_csv(f) for f in files]
	for i, d in enumerate(datasets):
		file_ids = [row['name'].rsplit('_', 1)[0] + '.wav' for id, row in d.iterrows()]
		d['file_id'] = file_ids

	frames_with_labels = [pd.merge(data, frame, on='file_id') for data, frame in zip(datasets, frames)]
	labels = [f['label'].values for f in frames_with_labels]
	# print('FRAMES WITH LABELS:', frames_with_labels[0])
	return labels


# Load data from CSV file
def load_data(files):
	frames = [pd.read_csv(f) for f in files]
	return frames


# plot training history
def plot_training_history(epochs, plottable, ylabel='', name=''):
	plt.clf()
	plt.xlabel('Epoch')
	plt.ylabel(ylabel)
	if len(plottable) == 1:
		plt.plot(np.arange(epochs), plottable[0], label='Loss')
	elif len(plottable) == 2:
		plt.plot(np.arange(epochs), plottable[0], label='Acc')
		plt.plot(np.arange(epochs), plottable[1], label='UAR')
	else:
		raise ValueError('plottable passed to plot function has incorrect dim.')
	plt.legend()
	plt.savefig('%s.png' % (name), bbox_inches='tight')
