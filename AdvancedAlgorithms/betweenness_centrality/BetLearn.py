import copy
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.callbacks import CallbackList
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


BATCH_SIZE = 16
max_bp_iter = 5  # neighbor propagation steps

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
aux_feat_dim = 4   # extra node features in the hidden layer in the decoder, [Dc,CI1,CI2,1]


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
    input_node_features = Input(shape=(node_feat_dim,), name='node_features')
    input_aux_features = Input(shape=(aux_feat_dim,), name='aux_features')
    input_n2n = Input(shape=(None,), sparse=True, name='n2n_sum')

    node_features = Dense(units=128)(input_node_features)
    node_features = LeakyReLU()(node_features)
    node_features = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalize_node_features')(node_features)

    n2n_features = DrBCRNN(units=128, repetitions=max_bp_iter, combine='gru')([input_n2n, node_features])
    n2n_features = Lambda(lambda x: tf.reduce_max(x, axis=-1), name='aggregate')(n2n_features)
    n2n_features = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='normalize_n2n')(n2n_features)

    all_features = Concatenate(axis=-1)([n2n_features, input_aux_features])
    top = Dense(64)(all_features)
    top = LeakyReLU()(top)
    out = Dense(1)(top)

    return Model(inputs=[input_node_features, input_aux_features, input_n2n], outputs=out, name='DrBC')


class DataGenerator(Sequence):
    def __init__(self, tag: str, graph_type: str, min_nodes: int, max_nodes: int, nb_graphs: int, graphs_per_batch: int, nb_batches: int,
                 include_idx_map: bool = False, random_samples: bool = True, log_betweenness: bool = True):
        self.utils = utils.py_Utils()
        self.graphs = graph.py_GSet()
        self.count: int = 0
        self.betweenness: List[float] = []
        self.tag: str = tag
        self.graph_type: str = graph_type
        self.min_nodes: int = min_nodes
        self.max_nodes: int = max_nodes
        self.nb_graphs: int = nb_graphs
        self.graphs_per_batch: int = graphs_per_batch
        self.nb_batches: int = nb_batches
        self.include_idx_map: bool = include_idx_map
        self.random_samples: bool = random_samples
        self.log_betweenness: bool = log_betweenness

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
        for _ in tqdm(range(self.nb_graphs), desc=f'{self.tag}: generating new graphs...'):
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


class EvaluateCallback(Callback):
    def __init__(self, data_generator, prepend_str: str = 'val_'):
        super().__init__()
        self.data_generator = data_generator
        self.prepend_str = prepend_str
        self.metrics = metrics.py_Metrics()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        logs = logs or {}
        epoch_logs = {
            f'{self.prepend_str}top0.01': [],
            f'{self.prepend_str}top0.05': [],
            f'{self.prepend_str}top0.1': [],
            f'{self.prepend_str}kendal': [],
        }
        for gid, (x, y, idx_map) in enumerate(self.data_generator):
            result = self.model.predict_on_batch(x=x).flatten()
            betw_predict = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0
                            for i, pred_betweenness in enumerate(result)]

            betw_label = self.data_generator.betweenness[gid]
            epoch_logs[f'{self.prepend_str}top0.01'].append(self.metrics.RankTopK(betw_label, betw_predict, 0.01))
            epoch_logs[f'{self.prepend_str}top0.05'].append(self.metrics.RankTopK(betw_label, betw_predict, 0.05))
            epoch_logs[f'{self.prepend_str}top0.1'].append(self.metrics.RankTopK(betw_label, betw_predict, 0.1))
            epoch_logs[f'{self.prepend_str}kendal'].append(self.metrics.RankKendal(betw_label, betw_predict))
        epoch_logs = {k: np.mean(val) for k, val in epoch_logs.items()}
        logs.update(epoch_logs)


class BetLearn:

    def __init__(self):
        self.experiment_path = Path('./experiments') / datetime.now().replace(microsecond=0).isoformat()
        self.model_save_path = self.experiment_path / 'models/'
        self.log_dir = self.experiment_path / 'logs/'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # graph types: 'powerlaw', 'erdos_renyi', 'powerlaw', 'small-world', 'barabasi_albert'
        self.train_generator = DataGenerator(tag='Train', graph_type='powerlaw', min_nodes=NUM_MIN, max_nodes=NUM_MAX, nb_graphs=n_train, graphs_per_batch=BATCH_SIZE, nb_batches=500, include_idx_map=False, random_samples=True, log_betweenness=True)
        self.valid_generator = DataGenerator(tag='Valid', graph_type='powerlaw', min_nodes=NUM_MIN, max_nodes=NUM_MAX, nb_graphs=n_valid, graphs_per_batch=1, nb_batches=n_valid, include_idx_map=True, random_samples=False, log_betweenness=False)

        self.model = create_drbc_model()
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=LossFunctionWrapper(pairwise_ranking_crossentropy_loss, reduction='none'))
        print(f'Logging experiments at: `{self.experiment_path.absolute()}`')

    def predict(self, gid):
        x, y, idx_map = self.valid_generator[gid]
        result = self.model.predict_on_batch(x=x).flatten()

        # idx_map[i] >= 0:  # corresponds to nodes with 0.0 betw_log value
        result_output = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0
                         for i, pred_betweenness in enumerate(result)]
        return result_output

    def train(self):
        """ functional API with model.fit doesn't support sparse tensors with the current implementation => we write the training loop ourselves """
        callbacks = CallbackList([
            EvaluateCallback(self.valid_generator, prepend_str='val_'),
            TensorBoard(self.log_dir, profile_batch=0),
            ModelCheckpoint(self.model_save_path / 'best.h5py', monitor='val_kendal', save_best_only=True, verbose=1, mode='max'),
            EarlyStopping(monitor='val_kendal', patience=5, mode='max', restore_best_weights=True),
        ], add_history=True, add_progbar=True, model=self.model, verbose=1, epochs=MAX_ITERATION, steps=len(self.train_generator))

        callbacks.on_train_begin()
        for epoch in range(MAX_ITERATION):
            if epoch % 5 == 0:
                self.train_generator.gen_new_graphs()
                self.valid_generator.gen_new_graphs()

            callbacks.on_epoch_begin(epoch)
            [c.on_train_begin() for c in callbacks]
            for batch, (x, y) in enumerate(self.train_generator):
                callbacks.on_train_batch_begin(batch)
                logs = self.model.train_on_batch(x, y, return_dict=True)
                callbacks.on_train_batch_end(batch, logs)

            epoch_logs = copy.copy(logs)
            callbacks.on_epoch_end(epoch, logs=epoch_logs)

        callbacks.on_train_end(copy.copy(epoch_logs))
        print(self.model.history.history)
