#!/usr/bin/env
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Train Model
def train_svm(X, y, parms):
    '''
    Train Support Vector Machine:
    - In this function we define and train an SVM.
    - We will use Sciki-Learn's function SVC
    - A detailed guide of Sciki-Learn can be found in: https://scikit-learn.org/stable/

    In Scikit-Learn we have several functions to define an SVM, these include: (https://scikit-learn.org/stable/modules/svm.html#svm)
     - SVC - SVM for classification, can be used with Linear, RBF and Polynomial Kernels, based on LibSVM
     - SVR - Similar to SVC, but for regression
     - LinearSVC - Faster implementation of the linear kernel, based on Liblinear
     - NuSVC - Similar to SVC, but the user can control the number of support vectors

    Since we will be doing classification we will use SVC. SVC requires (among others) the following parameters:
     - Kernel: 'poly', 'rbf', 'sigmoid', 'linear', among others
     - C: C works as a regularization parameter for the SVM.
          For larger values of C, a smaller margin will be accepted if the decision function is better at
          classifying all training points correctly.
          A lower C will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy.
         (Taken from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)

     - degree: Degree of the polynomial if the polynomial kernel is selected
     - gamma: Kernel coefficient for RBF

    Class Weight:
     - The dataset we are working with is unbalanced.
     - To help the classifier better model the under-represented class, we can provide the classifier with the
       'class_weight='balanced'' parameter which will adjust the weight of each sample inversely proportional
       to the number of samples of each class in the input data.

    Random State:
     - If we are selecting the best parameters for our classifier we want to ensure all its intrinsic parameters are
       initialized equally every time we train it with different parameters.
    '''

    if parms['kernel'] != 'linear':
        clf = SVC(kernel=parms['kernel'], C=parms['C'], degree=parms['d'], gamma=parms['g'], class_weight='balanced',
                  verbose=0, random_state=12345)
    else:
        clf = LinearSVC(C=parms['C'], class_weight='balanced', max_iter=10000, verbose=0, random_state=12345)

    clf.fit(X, y)
    return clf


# Compute Predictions and Metrics
def test_svm(X, y, clf):
    # After we have trained the model we can compute predictions on unseen data and use them to evaluate other metrics
    preds = clf.predict(X)

    # In this case we are using as metrics the average F1 Score, Precision and Recall.
    # If we want to learn better how the model is behaving for each class we can remove the "average" from the function's inputs
    prf = precision_recall_fscore_support(y, preds, labels=[0, 1], average='macro')
    accuracy = accuracy_score(y, preds)

    return prf, accuracy
