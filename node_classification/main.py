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
# import lightgbm as lgb
from sklearn import cross_validation
from sklearn import svm

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


def get_auc(model, node_label):
    def label_node_vec(row):
        try:
            row['vec'] = model.wv.syn0[model.wv.index2word.index(str(row['index']))]
        except:
            row['vec'] = 0
        return row            
    node_label = node_label.apply(label_node_vec, axis=1)
    all_data = pd.concat([node_label, node_label['vec'].apply(pd.Series)], axis=1)
    all_data['label'] = all_data['label'].apply(str)
    all_data_vec = all_data.drop(['index', 'label', 'vec'], axis=1)
    all_data_vec = all_data_vec.applymap(float)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(all_data_vec, all_data['label'], test_size=0.4, random_state=0)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print('end')


def get_dict_auc(model, node_label):
    node_vec = pd.DataFrame.from_dict(model, orient='index').reset_index()
    node_label_vec = node_vec.merge(node_label, on='index', how='inner')
    all_data_vec = node_label_vec.drop(['index', 'label'], axis=1)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(all_data_vec, node_label['label'], test_size=0.4, random_state=0)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def read_LINE_vectors(file_name):
    tmp_embedding = dict()
    file = open(file_name, 'r')
    for line in file.readlines()[1:]:
        numbers = line[:-2].split(' ')
        tmp_vector = list()
        for n in numbers[1:]:
            tmp_vector.append(float(n))
            tmp_embedding[numbers[0]] = np.asarray(tmp_vector)
    file.close()
    return tmp_embedding


def train_LINE_model(filename):
    preparation_command = 'LD_LIBRARY_PATH=/usr/local/lib\nexport LD_LIBRARY_PATH'
    os.system('python ../other_methods/OpenNE-master/src/main.py\
              --method line --input {}\
              --output ./LINE_tmp_embedding1.txt --graph-format adjlist\
              --representation-size 100 --order 1 --weighted --directed'.format(filename))
    os.system('python ../other_methods/OpenNE-master/src/main.py\
              --method line --input {}\
              --output ./LINE_tmp_embedding2.txt --graph-format adjlist\
              --representation-size 100 --order 2 --weighted --directed'.format(filename))
    first_order_embedding = read_LINE_vectors('./LINE_tmp_embedding1.txt')
    second_order_embedding = read_LINE_vectors('./LINE_tmp_embedding2.txt')
    final_embedding = dict()
    for node in first_order_embedding:
        final_embedding[node] = np.append(first_order_embedding[node], second_order_embedding[node])
    return final_embedding


args = parse_args()
file_name = args.input
edge_type = ('1', '2')
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)

node_label = pd.DataFrame()
node2vec_auc = 0
Deepwalk_auc = 0
LINE_auc = 0
for threshold in [0.5, 0.8, 0.99]:
    if 'ee' in file_name.split('.edges')[0]:
        node_label = pd.read_csv('./threshold_data/eclipse_{}.csv'.format(threshold))
        LINE_path = './data/LINE/ee_'
    else:
        node_label = pd.read_csv('./threshold_data/gnome_{}.csv'.format(threshold))
        LINE_path = './data/LINE/ge_'

    for et in edge_type:
        # node2vec_G = Random_walk.RWGraph(get_G_from_edges(edge_data_by_type[et]), args.directed, 2, 0.5)
        # node2vec_G.preprocess_transition_probs()
        # node2vec_walks = node2vec_G.simulate_walks(10, 10)
        # node2vec_model = train_deepwalk_embedding(node2vec_walks)
        # tmp_node2vec_auc = get_auc(node2vec_model, node_label)

        # Deepwalk_G = Random_walk.RWGraph(get_G_from_edges(edge_data_by_type[et]), args.directed, 1, 1)
        # Deepwalk_G.preprocess_transition_probs()
        # Deepwalk_walks = Deepwalk_G.simulate_walks(args.num_walks, 10)
        # Deepwalk_model = train_deepwalk_embedding(Deepwalk_walks)
        # tmp_Deepwalk_auc = get_auc(Deepwalk_model, node_label)

        LINE_file = LINE_path + str(et) + '.edges'
        LINE_model = train_LINE_model(LINE_file)
        tmp_LINE_auc = get_dict_auc(LINE_model, node_label)

        node2vec_auc += tmp_node2vec_auc
        Deepwalk_auc += tmp_Deepwalk_auc
        LINE_auc += tmp_LINE_auc

    # print('threshold:{}, node2vec_auc:{}'.format(threshold, round(node2vec_auc/2.0, 2)))
    # print('threshold:{}, Deepwalk_auc:{}'.format(threshold, round(Deepwalk_auc/2.0, 2)))
    print('threshold:{}, LINE_auc:{}'.format(threshold, round(LINE_auc/2.0, 2)))


PMNE_auc = 0
