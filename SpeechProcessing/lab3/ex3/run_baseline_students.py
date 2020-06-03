import random
from pathlib import Path
from sklearn.utils import class_weight

from nn_torch_functions import *

# Fix random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(12456)
random.seed(12345)
torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_nn(data_files, label_files, feature_set='egemaps'):
    # define training parameters:
    epochs = 50
    learning_rate = 0.001
    l2_decay = 0.05
    batch_size = 128
    dropout = 0.5  # Thanos

    # initialize dataset with the data files and label files
    dataset = SleepinessDataset(data_files, label_files)

    # Get number of classes and number of features from dataset
    n_classes = torch.unique(dataset.y).shape[0]
    n_features = dataset.X.shape[1]

    print(n_features)

    # initialize the model
    model = FeedforwardNetwork(n_classes, n_features, dropout)
    model = model.to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=2)

    # define loss function. The weights tensor corresponds to the weight we give
    # to each class. It corresponds to the inverse of the frequency of that class
    # in the training set. This is a strategy to deal with imbalanced datasets.
    class_weights = class_weight.compute_class_weight('balanced', np.unique(dataset.y), dataset.y.numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    # train the model
    model, train_mean_losses, valid_accs, valid_uar = train(dataset, model, optimizer, scheduler, criterion, batch_size, epochs)

    # evaluate on train set
    train_X, train_y = dataset.X, dataset.y
    train_acc, train_prf = evaluate(model, train_X, train_y)

    print('Final Train acc: %.4f' % train_acc)
    print('Final Train prf: ', train_prf)

    # evaluate on dev set
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    dev_acc, dev_prf = evaluate(model, dev_X, dev_y)

    print('Final dev acc: %.4f' % dev_acc)
    print('Final dev prf: ', dev_prf)

    # get predictions for test and dev set
    test_X = dataset.test_X
    predictions_dev = predict(model, dev_X)
    predictions_dev = predictions_dev.detach().cpu().numpy()

    predictions_test = predict(model, test_X)
    predictions_test = predictions_test.detach().cpu().numpy()

    pd.DataFrame({'file_id': [name.rsplit('_', 1)[0] + '.wav' for name in dataset.dev_names], 'predicted_label': predictions_dev, 'labels': dev_y}).to_csv(f'dev.{feature_set}.nn.csv', index=False)
    pd.DataFrame({'file_id': [name.rsplit('_', 1)[0] + '.wav' for name in dataset.test_names], 'predicted_label': predictions_test}).to_csv(f'test.{feature_set}.nn.csv', index=False)

    # save the model
    torch.save(model, f'nn_{feature_set}.pth')

    # plot training history
    plot_training_history(epochs, [train_mean_losses], ylabel='Loss', name=f'training-loss-{feature_set}')
    plot_training_history(epochs, [valid_accs, valid_uar], ylabel='Accuracy', name=f'validation-metrics-{feature_set}')


def main():
    directory = Path('../corpus/')  # Full path to your current folder
    feature_set = "is11"  # name of the folder with the feature set

    # Label files
    labels_train = directory / 'labels' / 'train_labels_transformed.csv'
    labels_devel = directory / 'labels' / 'dev_labels_tranformed.csv'

    label_files = [labels_train, labels_devel]

    # Data files is11_train_data.csv
    data_train = directory / f'{feature_set}_train.csv'
    data_devel = directory / f'{feature_set}_dev.csv'
    data_test = directory / f'{feature_set}_test.csv'
    data_files = [data_train, data_devel, data_test]

    run_nn(data_files, label_files, feature_set)


if __name__ == "__main__":
    main()
