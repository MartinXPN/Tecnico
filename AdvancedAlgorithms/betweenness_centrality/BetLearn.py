#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: fanchangjun
"""
import os
import pickle as cp
import random
import sys
import time

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Dense, LeakyReLU
from tensorflow.keras.models import Model
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


class BetLearn:

    def __init__(self):
        # init some parameters
        self.g_type = 'powerlaw'  # 'erdos_renyi', 'powerlaw', 'small-world', 'barabasi_albert'
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.reg_hidden = REG_HIDDEN
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.utils = utils.py_Utils()
        self.TrainBetwList = []
        self.TestBetwList = []
        self.ngraph_train = 0
        self.ngraph_test = 0

        self.metrics = metrics.py_Metrics()
        self.inputs = dict()

        self.model = create_drbc_model()
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=LossFunctionWrapper(pairwise_ranking_crossentropy_loss, reduction='none'))

    def gen_graph(self, num_min, num_max):
        cur_n = np.random.randint(num_max - num_min + 1) + num_min
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        self.ClearTrainGraphs()
        for i in tqdm(range(n_train)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)
            bc = self.utils.Betweenness(self.GenNetwork(g))
            bc_log = self.utils.bc_log
            self.TrainBetwList.append(bc_log)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()
        self.TrainBetwList = []
        self.TrainBetwRankList = []

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()
        self.TestBetwList = []

    def InsertGraph(self, g, is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        self.ClearTestGraphs()
        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            self.InsertGraph(g, is_test=True)
            bc = self.utils.Betweenness(self.GenNetwork(g))
            self.TestBetwList.append(bc)

    def SetupBatchGraph(self, g_list):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupBatchGraph(g_list)
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['node_feat'] = np.array(prepareBatchGraph.node_feat)
        self.inputs['aux_feat'] = np.array(prepareBatchGraph.aux_feat)
        self.inputs['pair_ids_src'] = np.array(prepareBatchGraph.pair_ids_src)
        self.inputs['pair_ids_tgt'] = np.array(prepareBatchGraph.pair_ids_tgt)
        assert (len(prepareBatchGraph.pair_ids_src) == len(prepareBatchGraph.pair_ids_tgt))
        return prepareBatchGraph.idx_map_list

    def Predict(self, g_list):
        idx_map_list = self.SetupBatchGraph(g_list)
        result = self.model.predict_on_batch(x=(self.inputs['node_feat'],
                                                self.inputs['aux_feat'],
                                                self.inputs['n2nsum_param']))
        result = result.flatten()

        idx_map = idx_map_list[0]
        # idx_map[i] >= 0:  # corresponds to nodes with 0.0 betw_log value
        result_output = [np.power(10, -pred_betweenness) if idx_map[i] >= 0 else 0 for i, pred_betweenness in enumerate(result)]
        return result_output

    def Fit(self):
        g_list, id_list = self.TrainSet.Sample_Batch(BATCH_SIZE)

        Betw_Label_List = []
        for id in id_list:
            Betw_Label_List += self.TrainBetwList[id]
        label = np.array(Betw_Label_List)
        self.inputs['label'] = label
        self.SetupBatchGraph(g_list)

        batch_size = len(self.inputs['label'])
        y_true = np.concatenate([
            np.reshape(self.inputs['label'], (batch_size, 1)),
            np.reshape(self.inputs['pair_ids_src'], (batch_size, -1)),
            np.reshape(self.inputs['pair_ids_tgt'], (batch_size, -1)),
        ], axis=-1)

        loss = self.model.train_on_batch(x=[self.inputs['node_feat'], self.inputs['aux_feat'], self.inputs['n2nsum_param']],
                                         y=y_true)
        return loss / len(g_list)

    def Train(self):
        self.PrepareValidData()
        self.gen_new_graphs(NUM_MIN, NUM_MAX)

        save_dir = './models'
        VCFile = '%s/ValidValue.csv' % (save_dir)
        f_out = open(VCFile, 'w')
        for iter in range(MAX_ITERATION):
            TrainLoss = self.Fit()
            start = time.clock()
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
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
        g_list = [self.TestSet.Get(gid)]
        start = time.time()
        betw_predict = self.Predict(g_list)
        end = time.time()
        betw_label = self.TestBetwList[gid]

        run_time = end - start
        topk = self.metrics.RankTopK(betw_label, betw_predict, 0.01)
        kendal = self.metrics.RankKendal(betw_label, betw_predict)
        return run_time, topk, kendal

    def findModel(self):
        VCFile = './models/ValidValue.csv'
        vc_list = []
        EarlyStop_start = 2
        EarlyStop_length = 1
        num_line = 0
        for line in open(VCFile):
            data = float(line.split(',')[0].strip(','))  # 0:topK; 1:kendal
            vc_list.append(data)
            num_line += 1
            if num_line > EarlyStop_start and data < np.mean(vc_list[-(EarlyStop_length + 1):-1]):
                best_vc = num_line
                break
        best_model_iter = 500 * best_vc
        best_model = './models/nrange_iter_%d.ckpt' % (best_model_iter)
        return best_model

    def EvaluateSynData(self, data_test, model_file=None):  # test synthetic data
        if model_file == None:  # if user do not specify the model_file
            model_file = self.findModel()
        print('The best model is :%s' % (model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
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
        self.LoadModel(model_file)
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

    def GenNetwork(self, g):  # networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B)
