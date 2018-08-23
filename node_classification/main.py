# This python file is used to reproduce our link prediction experiment
# Author: Hongming ZHANG, HKUST KnowComp Group

from sklearn.metrics import roc_auc_score
import math
import subprocess
import Node2Vec_LayerSelect
from gensim.models import Word2Vec
import argparse
from MNE import *
import pandas as pd
import os
import lightgbm as lgb

def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 100.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', type=int, default=10,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def train_deepwalk_embedding(walks, iteration=None):
    if iteration is None:
        iteration = 100
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=iteration)
    return model


args = parse_args()
file_name = args.input
edge_type = ('1', '2')
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)

node_label = pd.DataFrame()
for threshold in [0.5, 0.8, 0.99]:
    if 'ee' in file_name.split('.edges')[0]:
        node_label = pd.read_csv('./threshold_data/eclipse_{}.csv'.format(threshold))
    else:
        node_label = pd.read_csv('./threshold_data/gnome_{}.csv'.format(threshold))

    for et in edge_type:
        node2vec_G = Random_walk.RWGraph(get_G_from_edges(edge_data_by_type[et]), args.directed, 2, 0.5)
        node2vec_G.preprocess_transition_probs()
        node2vec_walks = node2vec_G.simulate_walks(10, 10)
        node2vec_model = train_deepwalk_embedding(node2vec_walks)

        def label_node_vec(row):
            try:
                row['vec'] = node2vec_model.wv.syn0[node2vec_model.wv.index2word.index(str(row['index']))]
            except:
                row['vec'] = 0
            return row            
        node_label = node_label.apply(label_node_vec, axis=1)
    all_data = pd.concat([node_label, node_label['vec'].apply(pd.Series)], axis=1)
    
    all_data_vec = all_data.drop(['index', 'label'], axis=1)
    train_data = lgb.Dataset(all_data_vec, label=all_data['label'])
    print(train_data)
    param = {}
    param['metric'] = 'auc'
    # clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
    # clf.fit(train_x,train_y,feature_name=features)
    # ypred = clf.predict_proba(test_x)[:,1]
    num_round = 2
    lgb.cv(param, train_data, num_round, nfold=2)
    print('end')