from functools import reduce
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

'''
To use this script your prediction and label files should have the following columns, in this order:
Prediction files: file_id predictions
Label files: file_id label
Files should include a header with the name of each column.

In this script you should change the path to your own prediction and label files.

You will have to compute the majory vote between the predictions you choose. In this script, we
are only loading two predictions, but you may use more.
'''


def main():
    # Load the predictions by speaker obtained with get_predictions_by_speaker.py
    preds_devel_1 = pd.read_csv('../ex3/dev.is11.nn.csv');              preds_devel_1.columns = ['file_id', 'predictions_1', 'label']
    preds_devel_2 = pd.read_csv('../ex3/dev.egemaps.nn.csv');           preds_devel_2.columns = ['file_id', 'predictions_2', 'label']
    preds_devel_3 = pd.read_csv('../ex2/dev.result.svm.egemaps.csv');   preds_devel_3.columns = ['file_id', 'predictions_3', 'label']
    preds_devel_4 = pd.read_csv('../ex2/dev.result.svm.is11.csv');      preds_devel_4.columns = ['file_id', 'predictions_4', 'label']

    preds_test_1 = pd.read_csv('../ex3/test.egemaps.nn.csv');           preds_test_1.columns = ['file_id', 'predictions_1']
    preds_test_2 = pd.read_csv('../ex3/test.is11.nn.csv');              preds_test_2.columns = ['file_id', 'predictions_2']
    preds_test_3 = pd.read_csv('../ex2/test.result.svm.egemaps.csv');   preds_test_3.columns = ['file_id', 'predictions_3']
    preds_test_4 = pd.read_csv('../ex2/test.result.svm.is11.csv');      preds_test_4.columns = ['file_id', 'predictions_4']

    # Merge all predictions and labels
    devel = reduce(lambda left, right: pd.merge(left, right, on=['file_id', 'label']), [preds_devel_1, preds_devel_2, preds_devel_3, preds_devel_4])
    test = reduce(lambda left, right: pd.merge(left, right, on='file_id'), [preds_test_1, preds_test_2, preds_test_3, preds_test_4])

    # devel = pd.merge(pd.merge(preds_devel_1, preds_devel_2, on='file_id'), preds_devel_3, on='file_id')
    # test = pd.merge(preds_test_1, preds_test_2, on='file_id')
    print(test.head())

    # devel['mv'] = devel[['predictions_1', 'predictions_2', 'predictions_3', 'predictions_4']].mode(axis=1)[0]
    # test['mv'] = test[['predictions_1', 'predictions_2', 'predictions_3', 'predictions_4']].mode(axis=1)[0]

    # Since the results on dev where better for predictions_2', 'predictions_3', 'predictions_4 we chose to
    # stick with them instead of doing voting for all the models
    devel['mv'] = devel[['predictions_2', 'predictions_3', 'predictions_4']].mode(axis=1)[0]
    test['mv'] = test[['predictions_2', 'predictions_3', 'predictions_4']].mode(axis=1)[0]

    # Print out the results for each model and for the final combination
    print("Results for the Development Dataset")
    print(f"Results1: {precision_recall_fscore_support(devel.label.values, devel.predictions_1.values, labels=[0, 1], average='macro')}")
    print(f"Results2: {precision_recall_fscore_support(devel.label.values, devel.predictions_2.values, labels=[0, 1], average='macro')}")
    print(f"Results3: {precision_recall_fscore_support(devel.label.values, devel.predictions_3.values, labels=[0, 1], average='macro')}")
    print(f"Results4: {precision_recall_fscore_support(devel.label.values, devel.predictions_4.values, labels=[0, 1], average='macro')}")
    print(f"Results for majority vote {precision_recall_fscore_support(devel.label.values, devel.mv.values, labels=[0, 1], average='macro')}")

    devel[['file_id', 'mv']].to_csv('dev.result.fusion.csv', index=False)
    test[['file_id', 'mv']].to_csv('test.result.fusion.csv', index=False)


if __name__ == "__main__":
    main()
