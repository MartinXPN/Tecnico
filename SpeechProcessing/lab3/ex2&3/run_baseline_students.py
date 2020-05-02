#!/usr/bin/env
import pickle as pkl
from pathlib import Path

from tools import *

from nn_torch_functions import *
from svm_functions import *

import numpy as np
import random as rn
import torch

# Fix random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(12456)
rn.seed(12345)
torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_svm(data_files, label_files):
    '''
     For the functions load_data() and load_labels() to work correctly:
        - data_files should be a size 3 array with the path of the data file for the Training, Development and Test sets.
        - label_files should be a size 2 array with the path of the label file for the Training and Development sets.
        - Each data file should contain a matrix with shape (N_samples, N_features), with semicolon separated values.
        - Each label file should contain 2 columns, separated with a comma. One column should contain the id of the speaker
          and the second sould contain the label (0 or 1). The header corresponding to these two columns should be file_id,
          label. The file may contain other columns, but those will be ignored.
    '''
    # Load data and labels
    X_train, X_devel, X_test = load_data(data_files)
    y_train, y_devel = load_labels(label_files, [X_train, X_devel])
    print(len(X_train), len(y_train))

    train_names = X_train['name']
    dev_names = X_devel['name']
    test_names = X_test['name']
    X_train.drop(labels=['name', 'frameTime', 'file_id'], axis=1, inplace=True)
    X_devel.drop(labels=['name', 'frameTime', 'file_id'], axis=1, inplace=True)
    X_test.drop(labels=['name', 'frameTime'], axis=1, inplace=True)
    print(X_train)
    print(y_train)
    '''
    Data Pre-processing:
     - This is an important step when preparing data for a classifier.
     - Normalizing or transforming data can greatly help the classifier achieve better results and train faster.
     - Sklearn has a several preprocessing functions that can be used to this end:
     - https://scikit-learn.org/stable/modules/preprocessing.html

     - If you have not done the feature processing at Part 1 - now it is a good time to do it.
    '''
    X_train = preprocessing.scale(X_train)
    X_devel = preprocessing.scale(X_devel)
    X_test = preprocessing.scale(X_test)

    # Define Model Parameters
    params = {'kernel': 'rbf',
              'C': 1,
              'g': 1,
              'd': 3}

    # Train Model
    # Inspect the function train_svm at svm_functions.py and change class_weight
    print('Train the model...')
    model = train_svm(X_train, y_train, params)

    # Test the model: compute predictions and metrics for train and devel
    train_prf, train_accuracy = test_svm(X_train, y_train, model)
    print('train - accuracy: ', train_accuracy, 'prf:', train_prf)

    dev_prf, dev_accuracy = test_svm(X_devel, y_devel, model)
    print('dev - accuracy: ', dev_accuracy, 'prf:', dev_prf)

    # Compute predictions for dev and test data
    predictions_dev = model.predict(X_devel)
    predictions_test = model.predict(X_test)

    # Save test predictions
    pd.DataFrame({'file_id': [name.rsplit('_', 1)[0] + '.wav' for name in dev_names], 'predicted_label': predictions_dev, 'labels': y_devel}).to_csv('dev_predictions.csv', index=False)
    pd.DataFrame({'file_id': [name.rsplit('_', 1)[0] + '.wav' for name in test_names], 'predicted_label': predictions_test}).to_csv('test_predictions.csv', index=False)

    # Save Model - After we train a model we can save it for later use
    pkl.dump(model, open('svm_model.pkl', 'wb'))


def run_nn(data_files, label_files):
    # define training parameters:
    # epochs = 20
    # learning_rate = 0.001
    # l2_decay = 0
    # batch_size = 64
    # dropout = 0.1

    epochs = '''TODO'''
    learning_rate = '''TODO'''
    l2_decay = '''TODO'''
    batch_size = '''TODO'''
    dropout = '''TODO'''

    # define loss function. The weights tensor corresponds to the weight we give
    # to each class. It corresponds to the inverse of the frequency of that class
    # in the training set. This is a strategy to deal with imbalanced datasets.
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1, 1], dtype=torch.float).to(device))  # TODO: change class weights

    # initialize dataset with the data files and label files
    # dataset = SleepinessDataset(data_files, label_files)
    dataset = SleepinessDataset('''TODO''')

    # Get number of classes and number of features from dataset
    n_classes = torch.unique(dataset.y).shape[0]
    n_features = '''TODO'''

    # initialize the model
    model = FeedforwardNetwork('''TODO''')
    model = model.to(device)

    # get an optimizer
    # define the optimizer:
    optimizer = 'sgd'
    optims = optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    '''TODO - Try different optimizers: Adam, Adagrad, ... Full list can be found in pytorch's documentation'''
    optim_cls = optims[optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_decay)

    # train the model
    model, train_mean_losses, valid_accs, valid_uar = '''TODO'''

    # evaluate on train set
    train_X, train_y = dataset.X, dataset.y
    train_acc, train_prf = '''TODO'''

    print('Final Train acc: %.4f' % (train_acc))
    print('Final Train prf: ', train_prf)

    # evaluate on dev set
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    dev_acc, dev_prf = '''TODO'''

    print('Final dev acc: %.4f' % (dev_acc))
    print('Final dev prf: ', dev_prf)

    # get predictions for test and dev set
    test_X = dataset.test_X
    predictions_dev = '''TODO'''
    predictions_dev = predictions_dev.detach().cpu().numpy()

    predictions_test = ''' TODO '''
    predictions_test = predictions_test.detach().cpu().numpy()

    # Save test predictions
    # TODO
    # you may use the function save_predictions in tools.py

    # save the model
    torch.save(model, 'nn_model.pth')

    # plot training history
    plot_training_history(epochs, [train_mean_losses], ylabel='Loss', name='training-loss')
    plot_training_history(epochs, [valid_accs, valid_uar], ylabel='Accuracy', name='validation-metrics')


def main():
    directory = Path('/Users/martin/Desktop/tecnico/SpeechProcessing/lab3/corpus/')  # Full path to your current folder
    feature_set = "egemaps"  # name of the folder with the feature set

    # Label files
    labels_train = directory / 'labels' / 'train_labels_transformed.csv'
    labels_devel = directory / 'labels' / 'dev_labels_tranformed.csv'

    label_files = [labels_train, labels_devel]

    # Data files is11_train_data.csv
    data_train = directory / f'{feature_set}_train.csv'
    data_devel = directory / f'{feature_set}_dev.csv'
    data_test = directory / f'{feature_set}_test.csv'
    data_files = [data_train, data_devel, data_test]

    # Run SVM - PART 2
    run_svm(data_files, label_files)


# Run NN - PART 3
# run_nn(data_files, label_files)

if __name__ == "__main__":
    main()
