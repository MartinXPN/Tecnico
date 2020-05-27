import os
import pickle as cp
import random
import sys
import time
from typing import List

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense, LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.losses import LossFunctionWrapper
from tqdm import tqdm

import PrepareBatchGraph
import graph
import metrics
import utils
from layers import DrBCRNN

# For reproducibility
SEED = 42
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+


EMBEDDING_SIZE = 128  # embedding dimension
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
max_bp_iter = 5  # neighbor propagation steps

REG_HIDDEN = int(EMBEDDING_SIZE / 2)  # hidden dimension in the  MLP decoder
initialization_stddev = 0.01
NUM_MIN = 100  # minimum training scale (node set size)
NUM_MAX = 200  # maximum training scale (node set size)
MAX_ITERATION = 10000  # training iterations
n_train = 1000  # number of train graphs
n_valid = 100  # number of validation graphs
aggregatorID = 2  # how to aggregate node neighbors, 0:sum; 1:mean; 2:GCN(weighted sum)
combineID = 1  # how to combine self embedding and neighbor embedding,
# 0:structure2vec(add node feature and neighbor embedding)
# 1:graphsage(concatenation);
# 2:gru
JK = 1  # layer aggregation,
# #0: do not use each layer's embedding;
# aggregate each layer's embedding with:
# 1:max_pooling;
# 2:min_pooling;
# 3:mean_pooling;
# 4:LSTM with attention
node_feat_dim = 3  # initial node features, [Dc,1,1]
aux_feat_dim = 4  # extra node features in the hidden layer in the decoder, [Dc,CI1,CI2,1]


@tf.function
def pairwise_ranking_crossentropy_loss(y_true, y_pred):
    pred_betweenness = y_pred
    target_betweenness = tf.slice(y_true, begin=(0, 0), size=(-1, 1))
    src_ids = tf.cast(tf.reshape(tf.slice(y_true, begin=(0, 1), size=(-1, 5)), (-1,)), 'int32')
    tgt_ids = tf.cast(tf.reshape(tf.slice(y_true, begin=(0, 6), size=(-1, 5)), (-1,)), 'int32')

    labels = tf.nn.embedding_lookup(target_betweenness, src_ids) - tf.nn.embedding_lookup(target_betweenness, tgt_ids)
    preds = tf.nn.embedding_lookup(pred_betweenness, src_ids) - tf.nn.embedding_lookup(pred_betweenness, tgt_ids)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=tf.sigmoid(labels))
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss


def create_drbc_model():
    input_node_features = Input(shape=(3,))
    input_aux_features = Input(shape=(4,))
    input_n2n = Input(shape=(None,), sparse=True)
    normalize = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))

    node_features = Dense(units=128)(input_node_features)
    node_features = LeakyReLU()(node_features)
    node_features = normalize(node_features)

    n2n_features = DrBCRNN()([input_n2n, node_features])
    n2n_features = Lambda(lambda x: tf.reduce_max(x, axis=-1), name='aggregate')(n2n_features)
    n2n_features = normalize(n2n_features)

    all_features = Concatenate(axis=-1)([n2n_features, input_aux_features])
    top = Dense(64)(all_features)
    top = LeakyReLU()(top)
    out = Dense(1)(top)

    return Model(inputs=[input_node_features, input_aux_features, input_n2n], outputs=out, name='DrBC')


class DataGenerator(Sequence):
    def __init__(self, graph_type: str, min_nodes: int, max_nodes: int, nb_graphs: int, graphs_per_batch: int, nb_batches: int, include_idx_map: bool = False, random_samples: bool = True, log_betweenness: bool = True):
        self.utils = utils.py_Utils()
        self.graphs = graph.py_GSet()
        self.count: int = 0
        self.betweenness: List[float] = []
        self.graph_type: str = graph_type
        self.min_nodes: int = min_nodes
        self.max_nodes: int = max_nodes
        self.nb_graphs: int = nb_graphs
        self.graphs_per_batch: int = graphs_per_batch
        self.nb_batches: int = nb_batches
        self.include_idx_map: bool = include_idx_map
        self.random_samples: bool = random_samples
        self.log_betweenness: bool = log_betweenness

        self.gen_new_graphs()

    def __len__(self) -> int:
        return self.nb_batches

    def get_batch(self, graphs, ids: List[int]):
        label = []
        for i in ids:
            label += self.betweenness[i]
        label = np.array(label)

        batch_graph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        batch_graph.SetupBatchGraph(graphs)
        assert (len(batch_graph.pair_ids_src) == len(batch_graph.pair_ids_tgt))

        batch_size = len(label)
        x = [
            np.array(batch_graph.node_feat),
            np.array(batch_graph.aux_feat),
            batch_graph.n2nsum_param
        ]
        y = np.concatenate([
            np.reshape(label, (batch_size, 1)),
            np.reshape(batch_graph.pair_ids_src, (batch_size, -1)),
            np.reshape(batch_graph.pair_ids_tgt, (batch_size, -1)),
        ], axis=-1)
        return (x, y, batch_graph.idx_map_list[0]) if self.include_idx_map else (x, y)

    def __getitem__(self, index: int):
        if self.random_samples:
            g_list, id_list = self.graphs.Sample_Batch(self.graphs_per_batch)
            return self.get_batch(graphs=g_list, ids=id_list)
        return self.get_batch(graphs=[self.graphs.Get(index)], ids=[index])

    def on_epoch_end(self):
        self.clear()
        self.gen_new_graphs()

    @staticmethod
    def gen_network(g):  # networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            a = np.array(a)
            b = np.array(b)
        else:
            a = np.array([0])
            b = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), a, b)

    def gen_graph(self):
        cur_n = np.random.randint(self.min_nodes, self.max_nodes)
        if self.graph_type == 'erdos_renyi':        return nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.graph_type == 'small-world':      return nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.graph_type == 'barabasi_albert':  return nx.barabasi_albert_graph(n=cur_n, m=4)
        elif self.graph_type == 'powerlaw':         return nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        raise ValueError(f'{self.graph_type} graph type is not supported yet')

    def gen_new_graphs(self):
        self.clear()
        for _ in tqdm(range(self.nb_graphs), desc='generating new graphs...'):
            g = self.gen_graph()
            t = self.count
            self.count += 1
            self.graphs.InsertGraph(t, self.gen_network(g))

            bc = self.utils.Betweenness(self.gen_network(g))
            bc_log = self.utils.bc_log
            self.betweenness.append(bc_log if self.log_betweenness else bc)

    def clear(self):
        self.count = 0
        self.graphs.Clear()
        self.betweenness = []


class BetLearn:

    def __init__(self):
        # init some parameters
        self.g_type = 'powerlaw'  # 'erdos_renyi', 'powerlaw', 'small-world', 'barabasi_albert'
        self.metrics = metrics.py_Metrics()

        self.model = create_drbc_model()
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=LossFunctionWrapper(pairwise_ranking_crossentropy_loss, reduction='none'))

        self.train_generator = DataGenerator(graph_type=self.g_type, min_nodes=NUM_MIN, max_nodes=NUM_MAX, nb_graphs=n_train, graphs_per_batch=BATCH_SIZE, nb_batches=5000, include_idx_map=False, random_samples=True, log_betweenness=True)
        self.valid_generator = DataGenerator(graph_type=self.g_type, min_nodes=NUM_MIN, max_nodes=NUM_MAX, nb_graphs=n_valid, graphs_per_batch=BATCH_SIZE, nb_batches=1, include_idx_map=True, random_samples=False, log_betweenness=False)

    def Predict(self, gid):
        x, y, idx_map = self.valid_generator[gid]
        result = self.model.predict_on_batch(x=x).flatten()

        # idx_map[i] >= 0:  # corresponds to nodes with 0.0 betw_log value
        result_output = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0
                         for i, pred_betweenness in enumerate(result)]
        return result_output

    def Fit(self):
        x, y = self.train_generator[0]  # next batch
        loss = self.model.train_on_batch(x=x, y=y)
        return loss

    def Train(self):
        save_dir = './models'
        VCFile = '%s/ValidValue.csv' % (save_dir)
        f_out = open(VCFile, 'w')
        for iter in range(MAX_ITERATION):
            TrainLoss = self.Fit()
            start = time.clock()
            if iter and iter % 5000 == 0:
                self.train_generator.on_epoch_end()
                self.valid_generator.on_epoch_end()
            if iter % 500 == 0:
                if (iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac_topk, frac_kendal = 0.0, 0.0
                test_start = time.time()
                for idx in range(n_valid):
                    run_time, temp_topk, temp_kendal = self.Test(idx)
                    frac_topk += temp_topk / n_valid
                    frac_kendal += temp_kendal / n_valid
                test_end = time.time()
                f_out.write('%.6f, %.6f\n' % (frac_topk, frac_kendal))  # write vc into the file
                f_out.flush()
                print('\niter %d, Top0.01: %.6f, kendal: %.6f' % (iter, frac_topk, frac_kendal))
                print('testing %d graphs time: %.2fs' % (n_valid, test_end - test_start))
                N_end = time.clock()
                print('500 iterations total time: %.2fs' % (N_end - N_start))
                print('Training loss is %.4f' % TrainLoss)
                sys.stdout.flush()
                model_path = '%s/nrange_iter_%d_%d_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.model.save(model_path)
        f_out.close()

    def Test(self, gid):
        start = time.time()
        betw_predict = self.Predict(gid)
        end = time.time()
        betw_label = self.valid_generator.betweenness[gid]

        run_time = end - start
        topk = self.metrics.RankTopK(betw_label, betw_predict, 0.01)
        kendal = self.metrics.RankKendal(betw_label, betw_predict)
        return run_time, topk, kendal

    def EvaluateSynData(self, data_test, model_file):  # test synthetic data
        print('The best model is :%s' % (model_file))
        sys.stdout.flush()
        load_model(model_file)
        n_test = 100
        frac_run_time, frac_topk, frac_kendal = 0.0, 0.0, 0.0
        self.ClearTestGraphs()
        f = open(data_test, 'rb')
        ValidData = cp.load(f)
        TestGraphList = ValidData[0]
        self.TestBetwList = ValidData[1]
        for i in tqdm(range(n_test)):
            g = TestGraphList[i]
            self.InsertGraph(g, is_test=True)
            run_time, topk, kendal = self.test(i)
            frac_run_time += run_time / n_test
            frac_topk += topk / n_test
            frac_kendal += kendal / n_test
        print('\nRun_time, Top1%, Kendall tau: %.6f, %.6f, %.6f' % (frac_run_time, frac_topk, frac_kendal))
        return frac_run_time, frac_topk, frac_kendal

    def EvaluateRealData(self, model_file, data_test, label_file):  # test real data
        g = nx.read_weighted_edgelist(data_test)
        sys.stdout.flush()
        load_model(model_file)
        betw_label = []
        for line in open(label_file):
            betw_label.append(float(line.strip().split()[1]))
        self.TestBetwList.append(betw_label)
        start = time.time()
        self.InsertGraph(g, is_test=True)
        end = time.time()
        run_time = end - start
        g_list = [self.TestSet.Get(0)]
        start1 = time.time()
        betw_predict = self.Predict(g_list)
        end1 = time.time()
        betw_label = self.TestBetwList[0]
        run_time += end1 - start1
        top001 = self.metrics.RankTopK(betw_label, betw_predict, 0.01)
        top005 = self.metrics.RankTopK(betw_label, betw_predict, 0.05)
        top01 = self.metrics.RankTopK(betw_label, betw_predict, 0.1)
        kendal = self.metrics.RankKendal(betw_label, betw_predict)
        self.ClearTestGraphs()
        return top001, top005, top01, kendal, run_time
