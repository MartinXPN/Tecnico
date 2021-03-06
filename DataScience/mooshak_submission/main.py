import sys, pandas as pd, numpy as np, seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn import datasets, metrics, cluster, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from column_name import colnames
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import arange
from mlxtend.frequent_patterns import apriori, association_rules
import operator
import warnings
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
import xgboost as xgb

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

'''A: read arguments'''
args = sys.stdin.readline().rstrip('\n').split(' ')
n, source, task = int(args[0]), args[1], args[2]

'''B: read dataset'''
data, header = [], sys.stdin.readline().rstrip().split(',')
for i in range(n - 1):
    inputs = sys.stdin.readline().rstrip().split(',')
    if len(inputs) == len(colnames[source]):
        data.append(inputs)
data = pd.DataFrame(data, columns=colnames[source])
data = data.iloc[2:]
data = data.apply(pd.to_numeric)
print(data.head())
'''C: output results'''


def split(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.flatten(), test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate(df, split_dataset=split, knn=None, tree=None, rf=None, boost=None):
    def report(y_pred):
        return {
            'Accuracy': metrics.accuracy_score(y_test, y_pred),
            'ROC AUC': metrics.roc_auc_score(y_test, y_pred),
            #         'Confusion Matrix': metrics.confusion_matrix(y_test, y_pred),
            'Sensitivity': metrics.classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
        }

    X_train, X_test, y_train, y_test = split_dataset(df)
    res = {}

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    res['Naive Bayes'] = report(nb.predict(X_test))

    if knn is None:
        knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    res['KNN'] = report(knn.predict(X_test))

    if tree is None:
        tree = DecisionTreeClassifier(class_weight='balanced')
    tree.fit(X_train, y_train)
    res['Decision Tree'] = report(tree.predict(X_test))

    if rf is None:
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    rf.fit(X_train, y_train)
    res['Random Forest'] = report(rf.predict(X_test))

    if type(X_train) is not np.ndarray:   X_train = X_train.values
    if type(X_test) is not np.ndarray:    X_test = X_test.values

    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)
    if boost is None:
        boost = xgb.train({'eta': 0.3, 'max_depth': 5, 'objective': 'multi:softprob', 'num_class': 2}, D_train, 10)
    preds = boost.predict(D_test)
    res['xgboost'] = report(np.asarray([np.argmax(line) for line in preds]))

    return res


def oversampled_data_split(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.flatten(), test_size=test_size,
                                                        random_state=42)
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def oversampled_ADASYN_data_split(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.flatten(), test_size=test_size,
                                                        random_state=42)
    X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def undersampled_data_split(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.flatten(), test_size=test_size,
                                                        random_state=42)
    X_train, y_train = RepeatedEditedNearestNeighbours().fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


standard = data.copy()
normal = data.copy()
for c in data.columns[2:-1]:
    standard[c] = (data[c] - data[c].mean()) / data[c].std()
    normal[c] = (data[c] - data[c].mean()) / (data[c].max() - data[c].min())


def get_inter_class_important_column_names(df, correlation_threshold=0, plot_correlations=True):
    superclasses = list(dict.fromkeys([c.split('-')[0] for c in df.columns]))
    groups = {c: [] for c in superclasses}
    for c in df.columns:
        groups[c.split('-')[0]].append(c)

    important_columns = []
    for i, (group_name, columns) in enumerate(groups.items()):
        corr_matrix = df[columns].corr().abs()  # Calculate the correlation within group
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))  # Select upper triangle of correlation matrix
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        to_keep = list(set(columns) - set(to_drop))
        important_columns += to_keep

        # Plot the correlation
        if not plot_correlations or i == 0 or i == len(groups.values()) - 1 or len(columns) > 10:
            continue
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 30))
        axes[0].set_title(f'In-group correlation for {group_name}')
        sns.heatmap(corr_matrix, ax=axes[0], xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
                    annot=True, cmap='Blues', square=True)

        corr_matrix = df[to_keep].corr().abs()
        axes[1].set_title('Correlation between the kept variables')
        sns.heatmap(corr_matrix, ax=axes[1], xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
                    annot=True, cmap='Blues', square=True)

        fig.tight_layout()
        plt.show()
    return important_columns


def get_intra_class_important_column_names(df, correlation_threshold=0, plot_correlations=False):
    columns = df.columns
    corr_matrix = df[columns].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))  # Select upper triangle of correlation matrix
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    to_keep = list(set(columns) - set(to_drop) | set(['class']))
    return to_keep



def split2(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.flatten(), test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate2(df, split_dataset=split, knn=None, tree=None, rf=None, boost=None):
    def report(y_pred):
        return {
            'Accuracy': metrics.accuracy_score(y_test, y_pred),
            #             'ROC AUC': metrics.roc_auc_score(y_test, y_pred),
            #             'Confusion Matrix': metrics.confusion_matrix(y_test, y_pred),
            'Sensitivity': metrics.classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
        }

    X_train, X_test, y_train, y_test = split_dataset(df)
    res = {}

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    res['Naive Bayes'] = report(nb.predict(X_test))

    if knn is None:
        knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    res['KNN'] = report(knn.predict(X_test))

    if tree is None:
        tree = DecisionTreeClassifier(class_weight='balanced')
    tree.fit(X_train, y_train)
    res['Decision Tree'] = report(tree.predict(X_test))

    if rf is None:
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    rf.fit(X_train, y_train)
    res['Random Forest'] = report(rf.predict(X_test))
    if type(X_train) is not np.ndarray:
        X_train = X_train.values
    if type(X_test) is not np.ndarray:
        X_test = X_test.values

    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)
    if boost is None:
        boost = xgb.train({'eta': 0.3, 'max_depth': 5, 'objective': 'multi:softprob', 'num_class': 7}, D_train, 10)
    preds = boost.predict(D_test)
    res['xgboost'] = report(np.asarray([np.argmax(line) for line in preds]))

    return res


def show_progress2(evaluations, labels, metric='Sensitivity'):
    methods = evaluations[0].keys()
    evaluations = [[results[metric] for method, results in evaluation.items()] for evaluation in evaluations]
    fig, ax = plt.subplots()
    ind = np.arange(len(evaluations[0]))
    width = 0.2

    p = [ax.bar(ind + i * width, height=evaluation, width=width, bottom=0) for i, evaluation in enumerate(evaluations)]

    ax.set_title(f'{metric} Scores')
    ax.set_xticks(ind + width / len(evaluations) * (len(evaluations) - 1))
    ax.set_xticklabels(methods)

    ax.legend(labels, loc='lower right', fancybox=True, shadow=True)
    ax.autoscale_view()


def oversampled_data_split2(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.flatten(), test_size=0.3, random_state=42)
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def oversampled_ADASYN_data_split2(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.flatten(), test_size=0.3, random_state=42)
    X_train, y_train = ADASYN().fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def undersampled_data_split2(df, test_size=0.3):
    X = df.loc[:, ~df.columns.isin(['class'])]
    y = df.loc[:, df.columns.isin(['class'])]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.flatten(), test_size=0.3, random_state=42)
    X_train, y_train = RepeatedEditedNearestNeighbours().fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def get_important_column_names2(df, correlation_threshold=0, plot_correlations=False):
    columns = df.columns
    corr_matrix = df[columns].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))  # Select upper triangle of correlation matrix
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    to_keep = list(set(columns) - set(to_drop) | set(['class']))
    return to_keep

    if percentage:
        ax.set_ylim(0.0, 1.0)

    legend = []
    colors = {}
    for name, y in yvalues.items():
        [method, measure] = name.split('-')
        style = ':' if measure == 'Sensitivity' else '-'
        c = colors.get(method, None)
        p = ax.plot(xvalues, y, linestyle=style, c=c)
        colors[method] = p[-1].get_color()
        legend.append(name)

    if plot_legend:
        ax.legend(legend, loc='lower left', fancybox=True, shadow=True)


def evaluate_threshold_performance2(df, correlation_thresholds, get_importants=get_important_column_names2,
                                    title='Performance of different classifiers', xlabel='Correlation Threshold'):
    performance = {}

    for threshold in tqdm(correlation_thresholds):
        importants = get_importants(data, correlation_threshold=threshold, plot_correlations=False)
        importants = df[importants]

        res = evaluate(importants, split_dataset=SPLIT_DATASET)
        for method, measures in res.items():
            for measure, value in measures.items():
                if measure not in {'Accuracy', 'Sensitivity'}:
                    continue
                name = f'{method}-{measure}'
                if name not in performance:
                    performance[name] = []
                performance[name].append(value)


SPLIT_DATASET = split


def get_decision_importants2(df, nb_importants=6):
    X_train, X_test, y_train, y_test = SPLIT_DATASET(df)
    tree = DecisionTreeClassifier(class_weight='balanced')
    tree.fit(X_train, y_train)
    col2importance = dict(zip(df.columns, tree.feature_importances_))
    best = sorted(col2importance.items(), key=operator.itemgetter(1), reverse=True)[:nb_importants]

    return [name for name, score in best]


def compute_known_distributions2(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)' % (mean, sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    #     sigma, loc, scale = _stats.lognorm.fit(x_values)
    #     distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    #     loc, scale = _stats.expon.fit(x_values)
    #     distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    a, loc, scale = _stats.skewnorm.fit(x_values)
    distributions['SkewNorm(%.2f)' % a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions


if source == "CT":
    data['class'] = data['class'] - 1
    data.head()
    target_count = pd.value_counts(data['class'].values.flatten())
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    X_train, X_test, y_train, y_test = SPLIT_DATASET(data)
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    if task == "preprocessing":
        print(evaluate2(data))
        standard = data.copy()
        normal = data.copy()
        for c in data.columns[0:10]:
            standard[c] = (data[c] - data[c].mean()) / data[c].std()
            normal[c] = (data[c] - data[c].mean()) / (data[c].max() - data[c].min())
        standard.head()

        print(evaluate2(standard))
        print(evaluate2(normal))

        print("\nOversampled : ", evaluate2(data, split_dataset=oversampled_data_split2))
        # print("\nOversampled ADASYN: ", evaluate2(data, split_dataset=oversampled_ADASYN_data_split2))
        print("\nUndersampled : ", evaluate2(data, split_dataset=undersampled_data_split2))

        importants = get_important_column_names2(data, correlation_threshold=0.4, plot_correlations=False)
        importants = data[importants]
        importants.head()
        importants = get_decision_importants2(data, nb_importants=8)
        importants = data[importants]

    if task == "classification":
        print("---naives Bayes---")
        EVALUATION_METRIC = 'Sensitivity'
        clf = BernoulliNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(metrics.classification_report(y_test, y_pred))
        print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
        # print('ROC AUC:', metrics.roc_auc_score(y_test, y_pred))
        print('Sensitivity:', metrics.classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        sns.heatmap(cnf_mtx, annot=True, cmap='Blues', fmt='g')

        print("---KNN---")
        nvalues = [1, 3, 5, 9, 13]
        dist = ['manhattan', 'chebyshev', 'minkowski']
        best = (-1, {})

        performance = {}
        for d in dist:
            performance[f'{d}-Accuracy'], performance[f'{d}-Sensitivity'] = [], []
            for n in nvalues:
                knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                performance[f'{d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                #         performance[f'{d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                performance[f'{d}-Sensitivity'].append(
                    metrics.classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
                if best[0] < performance[f'{d}-{EVALUATION_METRIC}'][-1]:
                    best = (performance[f'{d}-{EVALUATION_METRIC}'][-1], {'n_neighbors': n, 'metric': d})
            acc = performance[f'{d}-Accuracy']
            #     auc = performance[f'{d}-ROC AUC']
            sens = performance[f'{d}-Sensitivity']
            print(f'[d={d}] Best acc: {max(acc)}, with nvalues: {nvalues[np.argmax(acc)]}')
            #     print(f'[d={d}] Best auc: {max(auc)}, with nvalues: {nvalues[np.argmax(auc)]}')
            print(f'[d={d}] Best sensitivity: {max(sens)}, with nvalues: {nvalues[np.argmax(sens)]}')
            print('---------------------------')

        knn_best = best
        print(f'Obtained best {EVALUATION_METRIC}: {best[0]} - with params: {best[1]}')

        print("--Decision Tree---")

        min_samples_leaf = np.array(range(1, 200, 25)) / 1000.
        print('Min samples leaf:', min_samples_leaf)
        max_depths = [3, 7, 15]
        criteria = ['entropy', 'gini']
        best = (-1, {})
        for k, f in enumerate(criteria):
            performance = {}
            for d in max_depths:
                performance[f'd={d}-Accuracy'], performance[f'd={d}-Sensitivity'] = [], []
                for n in min_samples_leaf:
                    tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f, class_weight='balanced')
                    tree.fit(X_train, y_train)
                    y_pred = tree.predict(X_test)
                    performance[f'd={d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                    #             performance[f'd={d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                    performance[f'd={d}-Sensitivity'].append(
                        metrics.classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
                    if best[0] < performance[f'd={d}-{EVALUATION_METRIC}'][-1]:
                        best = (performance[f'd={d}-{EVALUATION_METRIC}'][-1],
                                {'min_samples_leaf': n, 'max_depth': d, 'criterion': f, 'class_weight': 'balanced'})

                acc = performance[f'd={d}-Accuracy']
                #         auc = performance[f'd={d}-ROC AUC']
                sens = performance[f'd={d}-Sensitivity']

                print(
                    f'[criterion={f}, d={d}] Best acc: {max(acc)}, with min_samples_leaf: {min_samples_leaf[np.argmax(acc)]}')
                #         print(f'[criterion={f}, d={d}] Best auc: {max(auc)}, with min_samples_leaf: {min_samples_leaf[np.argmax(auc)]}')
                print(
                    f'[criterion={f}, d={d}] Best sensitivity: {max(sens)}, with min_samples_leaf: {min_samples_leaf[np.argmax(sens)]}')
                print('---------------------------')
        decision_best = best

        print("---Random Forest---")
        n_estimators = [5, 10, 50, 100, 150, 200]
        max_depths = [3, 7, 15]
        max_features = ['sqrt', 'log2']
        best = (-1, {})
        for k, f in enumerate(max_features):
            performance = {}

            for d in max_depths:
                performance[f'd={d}-Accuracy'], performance[f'd={d}-Sensitivity'] = [], []
                for n in n_estimators:
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, class_weight='balanced')
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    performance[f'd={d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                    #             performance[f'd={d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                    performance[f'd={d}-Sensitivity'].append(
                        metrics.classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
                    if best[0] < performance[f'd={d}-{EVALUATION_METRIC}'][-1]:
                        best = (performance[f'd={d}-{EVALUATION_METRIC}'][-1],
                                {'n_estimators': n, 'max_depth': d, 'max_features': f, 'class_weight': 'balanced'})

                acc = performance[f'd={d}-Accuracy']
                #         auc = performance[f'd={d}-ROC AUC']
                sens = performance[f'd={d}-Sensitivity']

                print(
                    f'[max_features={k}, d={d}] Best acc: {max(acc)}, with n_estimators: {n_estimators[np.argmax(acc)]}')
                #         print(f'[max_features={f}, d={d}] Best auc: {max(auc)}, with n_estimators: {n_estimators[np.argmax(auc)]}')
                print(
                    f'[max_features={f}, d={d}] Best sensitivity: {max(sens)}, with n_estimators: {n_estimators[np.argmax(sens)]}')
                print('---------------------------')
        random_best = best

        print("---XGBOOST---")
        D_train = xgb.DMatrix(X_train.values, label=y_train)
        D_test = xgb.DMatrix(X_test.values, label=y_test)
        nb_steps = list(range(5, 56, 10))
        max_depths = [3, 7, 15]
        best = (-1, {})

        performance = {}
        for d in max_depths:
            performance[f'd={d}-Accuracy'], performance[f'd={d}-Sensitivity'] = [], []
            for n in nb_steps:
                param = {
                    'eta': 0.3,
                    'max_depth': d,
                    'objective': 'multi:softprob',
                    'num_class': 7
                }
                model = xgb.train(param, D_train, n)
                preds = model.predict(D_test)
                y_pred = np.asarray([np.argmax(line) for line in preds])

                performance[f'd={d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                #         performance[f'd={d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                performance[f'd={d}-Sensitivity'].append(
                    metrics.classification_report(y_test, y_pred, output_dict=True)['1']['recall'])
                if best[0] < performance[f'd={d}-{EVALUATION_METRIC}'][-1]:
                    best = (performance[f'd={d}-{EVALUATION_METRIC}'][-1], {'num_boost_round': n, 'params': param})

            acc = performance[f'd={d}-Accuracy']
            #     auc = performance[f'd={d}-ROC AUC']
            sens = performance[f'd={d}-Sensitivity']
            print(f'[d={d}] Best acc: {max(acc)}, with nb_steps: {nb_steps[np.argmax(acc)]}')
            #     print(f'[d={d}] Best auc: {max(auc)}, with nb_steps: {nb_steps[np.argmax(auc)]}')
            print(f'[d={d}] Best sensitivity: {max(sens)}, with nb_steps: {nb_steps[np.argmax(sens)]}')
            print('---------------------------')

        xg_best = best
        print(f'Obtained best {EVALUATION_METRIC}: {best[0]} - with params: {best[1]}')

        boost = xgb.train(dtrain=D_train, **xg_best[1])
        res = evaluate2(data, split_dataset=SPLIT_DATASET,
                       knn=KNeighborsClassifier(**knn_best[1]),
                       tree=DecisionTreeClassifier(**decision_best[1]),
                       rf=RandomForestClassifier(**random_best[1]),
                       boost=boost)

        print("Best Model: ", res)

    if task == "unsupervised":
        selected_columns = data.columns
        selected_columns = set(selected_columns)
        selected_columns.remove('class')

        newdf = data[selected_columns].copy()
        for col in newdf:
            if not col.startswith('y'):
                newdf[col] = pd.cut(newdf[col], 3, labels=['0', '1', '2'])
        dummylist = []
        for att in newdf:
            if att.startswith('y'):
                dummylist.append(newdf[[att]])
            else:
                dummylist.append(pd.get_dummies(newdf[[att]]))
        dummified_df = pd.concat(dummylist, axis=1)
        minpatterns = 100
        min_support = 0.45
        assoc = {}

        while min_support > 0:
            min_support *= 0.99

            frequent_itemsets = apriori(dummified_df, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
            rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
            res = rules[(rules['antecedent_len'] >= 3)]
            assoc[min_support] = len(res)
            if len(res) >= minpatterns:
                print(res)
                break

        print(f'Min support: {min_support}')

        errors = {'Sum of squares': []}
        ks = list(range(1, 16, 2))
        for k in ks:
            X = data.loc[:, ~data.columns.isin(['class'])]
            kmeans_model = cluster.KMeans(n_clusters=k, random_state=1).fit(X)
            errors['Sum of squares'].append(kmeans_model.inertia_)

        pca = PCA(n_components=2)
        pca.fit(data.loc[:, ~data.columns.isin(['class'])])

        print("Components (eigenvectors):\n", pca.components_)
        print("Explained variance (eigenvalues) =", pca.explained_variance_)
        print("Explained variance (ratio) =", pca.explained_variance_ratio_)

        X_pca = pca.transform(X)
        print(X_pca.shape)
        pca = PCA(n_components=10)
        pca.fit(data.loc[:, ~data.columns.isin(['class'])])

        x = pca.explained_variance_
        print("---PCA---")
        print(np.arange(len(x)), x)

###GLOBAL ANALYSIS
if source == "PD":

    importants = get_inter_class_important_column_names(data, correlation_threshold=0.4, plot_correlations=False)
    importants = data[importants]

    X_train, X_test, y_train, y_test = undersampled_data_split(data)
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    if task == "preprocessing":
        print("ccc")
        print(evaluate(data))
        print(evaluate(data, split_dataset=oversampled_data_split))
        # print(evaluate(data, split_dataset=oversampled_ADASYN_data_split))
        print(evaluate(data, split_dataset=undersampled_data_split))

        print("\nBEST Result: ", evaluate(data, split_dataset=oversampled_data_split,
                                          knn=KNeighborsClassifier(n_neighbors=1, metric='manhattan'),
                                          tree=DecisionTreeClassifier(min_samples_leaf=0.001, max_depth=7,
                                                                      criterion='entropy', class_weight='balanced'),
                                          rf=RandomForestClassifier(n_estimators=50, max_depth=15, max_features='log2',
                                                                    class_weight='balanced')))
        quit()

    ### 1. Naive bayes
    if task == "classification":
        EVALUATION_METRIC = 'ROC AUC'

        print('------Naive bayes---------')
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
        print('ROC AUC:', metrics.roc_auc_score(y_test, y_pred))
        print('Sensitivity:', metrics.recall_score(y_test, y_pred))
        cnf_mtx = metrics.confusion_matrix(y_test, y_pred)
        sns.heatmap(cnf_mtx, annot=True, cmap='Blues', fmt='g')
        ### KNN works ##
        nvalues = [1, 3, 5, 7, 9, 11, 13]
        dist = ['manhattan', 'euclidean', 'chebyshev', 'minkowski']

        print('------KNN---------')
        best = (-1, {})
        performance = {}
        for d in dist:
            performance[f'{d}-Accuracy'], performance[f'{d}-ROC AUC'], performance[f'{d}-Sensitivity'] = [], [], []
            for n in nvalues:
                knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                performance[f'{d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                performance[f'{d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                performance[f'{d}-Sensitivity'].append(metrics.recall_score(y_test, y_pred))
                if best[0] < performance[f'{d}-{EVALUATION_METRIC}'][-1]:
                    best = (performance[f'{d}-{EVALUATION_METRIC}'][-1], {'n_neighbors': n, 'metric': d})
            #     acc = performance[f'{d}-Accuracy']
            auc = performance[f'{d}-ROC AUC']
            sens = performance[f'{d}-Sensitivity']
            #     print(f'[d={d}] Best acc: {max(acc)}, with nvalues: {nvalues[np.argmax(acc)]}')
            print(f'[d={d}] Best auc: {max(auc)}, with nvalues: {nvalues[np.argmax(auc)]}')
            print(f'[d={d}] Best sensitivity: {max(sens)}, with nvalues: {nvalues[np.argmax(sens)]}')
            print('---------------------------')
        knn_best = best
        print(f'Obtained best {EVALUATION_METRIC}: {best[0]} - with params: {best[1]}')

        print("------------Decision Tree--------------")
        min_samples_leaf = np.array(range(1, 200, 10)) / 1000.
        print('Min samples leaf:', min_samples_leaf)
        max_depths = [3, 5, 7, 10, 15]
        criteria = ['entropy', 'gini']
        best = (-1, {})

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
        for k, f in enumerate(criteria):
            performance = {}
            for d in max_depths:
                performance[f'd={d}-Accuracy'], performance[f'd={d}-ROC AUC'], performance[
                    f'd={d}-Sensitivity'] = [], [], []
                for n in min_samples_leaf:
                    tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f, class_weight='balanced')
                    tree.fit(X_train, y_train)
                    y_pred = tree.predict(X_test)
                    performance[f'd={d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                    performance[f'd={d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                    performance[f'd={d}-Sensitivity'].append(metrics.recall_score(y_test, y_pred))
                    if best[0] < performance[f'd={d}-{EVALUATION_METRIC}'][-1]:
                        best = (performance[f'd={d}-{EVALUATION_METRIC}'][-1],
                                {'min_samples_leaf': n, 'max_depth': d, 'criterion': f, 'class_weight': 'balanced'})

                acc = performance[f'd={d}-Accuracy']
                auc = performance[f'd={d}-ROC AUC']
                sens = performance[f'd={d}-Sensitivity']

                print(
                    f'[criterion={f}, d={d}] Best acc: {max(acc)}, with min_samples_leaf: {min_samples_leaf[np.argmax(acc)]}')
                print(
                    f'[criterion={f}, d={d}] Best auc: {max(auc)}, with min_samples_leaf: {min_samples_leaf[np.argmax(auc)]}')
                print(
                    f'[criterion={f}, d={d}] Best sensitivity: {max(sens)}, with min_samples_leaf: {min_samples_leaf[np.argmax(sens)]}')
                print('---------------------------')
                del performance[f'd={d}-Accuracy']



        decision_best = best
        # Random forest
        print("---Random Forest---")
        n_estimators = [5, 10, 25, 50, 100, 150]
        max_depths = [5, 10, 15]
        max_features = ['sqrt', 'log2']
        best = (-1, {})
        for k, f in enumerate(max_features):
            performance = {}

            for d in max_depths:
                performance[f'd={d}-Accuracy'], performance[f'd={d}-ROC AUC'], performance[
                    f'd={d}-Sensitivity'] = [], [], []
                for n in n_estimators:
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, class_weight='balanced')
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    performance[f'd={d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                    performance[f'd={d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                    performance[f'd={d}-Sensitivity'].append(metrics.recall_score(y_test, y_pred))
                    if best[0] < performance[f'd={d}-{EVALUATION_METRIC}'][-1]:
                        best = (performance[f'd={d}-{EVALUATION_METRIC}'][-1],
                                {'n_estimators': n, 'max_depth': d, 'max_features': f, 'class_weight': 'balanced'})

                acc = performance[f'd={d}-Accuracy']
                auc = performance[f'd={d}-ROC AUC']
                sens = performance[f'd={d}-Sensitivity']

                print(
                    f'[max_features={f}, d={d}] Best acc: {max(acc)}, with n_estimators: {n_estimators[np.argmax(acc)]}')
                print(
                    f'[max_features={f}, d={d}] Best auc: {max(auc)}, with n_estimators: {n_estimators[np.argmax(auc)]}')
                print(
                    f'[max_features={f}, d={d}] Best sensitivity: {max(sens)}, with n_estimators: {n_estimators[np.argmax(sens)]}')
                print('---------------------------')
                del performance[f'd={d}-Accuracy']

        random_best = best
        print(f'Obtained best {EVALUATION_METRIC}: {best[0]} - with params: {best[1]}')
        print("---XGBOOST---")
        D_train = xgb.DMatrix(X_train, label=y_train)
        D_test = xgb.DMatrix(X_test, label=y_test)
        nb_steps = list(range(5, 31, 5))
        max_depths = [3, 5, 7, 10, 15]
        best = (-1, {})

        performance = {}
        for d in max_depths:
            performance[f'd={d}-Accuracy'], performance[f'd={d}-ROC AUC'], performance[
                f'd={d}-Sensitivity'] = [], [], []
            for n in nb_steps:
                param = {
                    'eta': 0.3,
                    'max_depth': d,
                    'objective': 'multi:softprob',
                    'num_class': 2
                }
                model = xgb.train(param, D_train, n)
                preds = model.predict(D_test)
                y_pred = np.asarray([np.argmax(line) for line in preds])

                performance[f'd={d}-Accuracy'].append(metrics.accuracy_score(y_test, y_pred))
                performance[f'd={d}-ROC AUC'].append(metrics.roc_auc_score(y_test, y_pred))
                performance[f'd={d}-Sensitivity'].append(metrics.recall_score(y_test, y_pred))
                if best[0] < performance[f'd={d}-{EVALUATION_METRIC}'][-1]:
                    best = (performance[f'd={d}-{EVALUATION_METRIC}'][-1], {'num_boost_round': n, 'params': param})

            acc = performance[f'd={d}-Accuracy']
            auc = performance[f'd={d}-ROC AUC']
            sens = performance[f'd={d}-Sensitivity']
            print(f'[d={d}] Best acc: {max(acc)}, with nb_steps: {nb_steps[np.argmax(acc)]}')
            print(f'[d={d}] Best auc: {max(auc)}, with nb_steps: {nb_steps[np.argmax(auc)]}')
            print(f'[d={d}] Best sensitivity: {max(sens)}, with nb_steps: {nb_steps[np.argmax(sens)]}')
            print('---------------------------')
            del performance[f'd={d}-Accuracy']

        xg_best = best
        print(f'Obtained best {EVALUATION_METRIC}: {best[0]} - with params: {best[1]}')

        print("BEST")
        boost = xgb.train(dtrain=D_train, **xg_best[1])
        res = evaluate(data, split_dataset=SPLIT_DATASET,
                       knn=KNeighborsClassifier(**knn_best[1]),
                       tree=DecisionTreeClassifier(**decision_best[1]),
                       rf=RandomForestClassifier(**random_best[1]),
                       boost=boost)

        print(res)




    if task == "unsupervised":
        selected_columns = get_inter_class_important_column_names(data, correlation_threshold=0.1,
                                                                  plot_correlations=False)
        selected_columns = set(selected_columns + ['-gender'])
        selected_columns.remove('-id')
        selected_columns.remove('class')

        newdf = data[selected_columns].copy()
        for col in newdf:
            if col not in ['-gender']:
                newdf[col] = pd.cut(newdf[col], 3, labels=['0', '1', '2'])
        minpatterns = 500
        min_support = 1.
        assoc = {}
        dummylist = []
        for att in newdf:
            if att in ['-gender']:
                newdf[att] = newdf[att].astype('category')
            dummylist.append(pd.get_dummies(newdf[[att]]))
        dummified_df = pd.concat(dummylist, axis=1)
        dummified_df.head(5)

        minpatterns = 100
        min_support = 0.9
        assoc = {}

        while min_support > 0:
            min_support *= 0.99

            frequent_itemsets = apriori(dummified_df, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
            rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
            res = rules[(rules['antecedent_len'] >= 3)]
            assoc[min_support] = len(res)
            if len(res) >= minpatterns:
                print(res.head())
                break

        print(f'Min support pd.cut: {min_support}')
        print('-----------------------------------')

        selected_columns = get_inter_class_important_column_names(data, correlation_threshold=0.1,
                                                                  plot_correlations=False)
        selected_columns = set(selected_columns + ['-gender'])
        selected_columns.remove('-id')
        selected_columns.remove('class')

        newdf = data[selected_columns].copy()
        for col in newdf:
            if col not in ['-gender']:
                newdf[col] = pd.qcut(newdf[col].rank(method='first'), 3, labels=['0', '1', '2'], duplicates='drop')

        selected_columns = get_inter_class_important_column_names(data, correlation_threshold=0.1,
                                                                  plot_correlations=False)
        selected_columns = set(selected_columns + ['-gender'])
        selected_columns.remove('-id')
        selected_columns.remove('class')

        newdf = data[selected_columns].copy()
        for col in newdf:
            if col not in ['-gender']:
                newdf[col] = pd.qcut(newdf[col].rank(method='first'), 3, labels=['0', '1', '2'], duplicates='drop')
        newdf.head(5)

        minpatterns = 100
        min_support = 0.5
        assoc = {}

        while min_support > 0:
            min_support *= 0.99

            frequent_itemsets = apriori(dummified_df, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
            rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
            res = rules[(rules['antecedent_len'] >= 3)]
            assoc[min_support] = len(res)
            if len(res) >= minpatterns:
                print(res.head())
                break

        print(f'Min support pd.qcut: {min_support}')

        errors = {'Sum of squares': []}
        ks = list(range(1, 33))
        for k in ks:
            X = data.loc[:, ~data.columns.isin(['class'])]
            kmeans_model = cluster.KMeans(n_clusters=k, random_state=1).fit(X)
            errors['Sum of squares'].append(kmeans_model.inertia_)

        kmeans_model = cluster.KMeans(n_clusters=3, random_state=1).fit(data.loc[:, ~data.columns.isin(['class'])])
        matrix = contingency_matrix(data['class'].values, kmeans_model.labels_)
        print('contingency_matrix for k = 3')
        print(matrix)

        errors = {'Sum of squares': []}
        ks = list(range(1, 33))
        for k in ks:
            X = data.loc[:, ~data.columns.isin(['class'])]
            kmeans_model = cluster.KMeans(n_clusters=k, random_state=1).fit(X)
            errors['Sum of squares'].append(kmeans_model.inertia_)

        distances = ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
        epss = arange(1, 7, 0.5)
        print('eps:', epss)
        print(errors)

        performance = {}
    #   for d in distances:
    #      performance[d] = []
    #     for e in epss:
    #        X = data.loc[:, ~data.columns.isin(['class'])]
    #       db = DBSCAN(eps=e, min_samples=2, metric=d)
    #       labels = db.fit_predict(X)
    #      performance[d].append(metrics.silhouette_score(X, labels))
