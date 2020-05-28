import pickle as cp
import time

import fire
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tqdm import trange, tqdm

import metrics
from betlearn import DataGenerator, EvaluateCallback

metrics = metrics.py_Metrics()


def evaluate_synthetic_data(data_test, model_path):
    """ This function is most probably wrong because the synthetic data is one file per graph and score """
    model = load_model(model_path, custom_objects={'tf': tf}, compile=False)
    data = DataGenerator(tag='Synthetic', include_idx_map=True, random_samples=False, compute_betweenness=False)
    evaluate = EvaluateCallback(data, prepend_str='')
    evaluate.set_model(model)

    with open(data_test, 'rb') as f:
        valid_data = cp.load(f)
    graph_list = valid_data[0]
    betweenness_list = valid_data[1]

    for i in trange(100):
        g = graph_list[i]
        data.add_graph(g)
    data.betweenness = betweenness_list

    res = {}
    evaluate.on_epoch_end(0, res)
    print(res)


def evaluate_real_data(model_path, data_test, label_file):
    model: Model = load_model(model_path, custom_objects={'tf': tf}, compile=False)
    data = DataGenerator(tag='RealData', include_idx_map=True, random_samples=False, compute_betweenness=False)
    evaluate = EvaluateCallback(data, prepend_str='')
    evaluate.set_model(model)

    exact_betweenness = []
    with open(label_file) as f:
        for line in tqdm(f):
            exact_betweenness.append(float(line.strip().split()[1]))

    start = time.time()
    g = nx.read_weighted_edgelist(data_test)
    data.add_graph(g)
    data.betweenness = [exact_betweenness]
    end = time.time()

    run_time = end - start
    start1 = time.time()
    x, y, idx_map = data[0]
    result = model.predict_on_batch(x=x).flatten()
    betw_predict = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0
                    for i, pred_betweenness in enumerate(tqdm(result))]
    data.clear()
    end1 = time.time()

    run_time += end1 - start1
    res = {
        'top001': metrics.RankTopK(exact_betweenness, betw_predict, 0.01),
        'top005': metrics.RankTopK(exact_betweenness, betw_predict, 0.05),
        'top01': metrics.RankTopK(exact_betweenness, betw_predict, 0.1),
        'kendal': metrics.RankKendal(exact_betweenness, betw_predict),
        'run_time': run_time,
    }

    return res


if __name__ == '__main__':
    fire.Fire({
        'synthetic': evaluate_synthetic_data,
        'real': evaluate_real_data,
    })
