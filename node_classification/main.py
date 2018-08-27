# This python file is used to reproduce our link prediction experiment
# Author: Hongming ZHANG, HKUST KnowComp Group

from sklearn.metrics import roc_auc_score
import math
import subprocess
import Node2Vec_LayerSelect
import Convince_Graph
import Convince_Multi_Graph

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


def classify_model(all_data_vec, all_data):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(all_data_vec, all_data['label'], test_size=0.4, random_state=0)
    clf = svm.SVC(kernel='linear', class_weight='balanced', C=1).fit(X_train, y_train)
    return clf.score(X_test, y_test)


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

    return classify_model(all_data_vec, all_data)


def get_dict_auc(model, node_label):
    node_vec = pd.DataFrame.from_dict(model, orient='index').reset_index()
    node_vec['index'] = node_vec['index'].apply(str)
    node_label['index'] = node_label['index'].apply(str)
    node_label_vec = node_vec.merge(node_label, on='index', how='inner')
    all_data_vec = node_label_vec.drop(['index', 'label'], axis=1)

    return classify_model(all_data_vec, node_label)


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


def get_PMNE_two_auc(model_one, model_two, node_label):
    def label_node_vec(row):
        try:
            row['vec'] = model_one.wv.syn0[model_one.wv.index2word.index(str(row['index']))] + model_two.wv.syn0[model_two.wv.index2word.index(str(row['index']))]
        except:
            row['vec'] = 0
        return row            
    node_label = node_label.apply(label_node_vec, axis=1)
    all_data = pd.concat([node_label, node_label['vec'].apply(pd.Series)], axis=1)
    all_data['label'] = all_data['label'].apply(str)
    all_data_vec = all_data.drop(['index', 'label', 'vec'], axis=1)
    all_data_vec = all_data_vec.applymap(float)

    return classify_model(all_data_vec, all_data)


def merge_PMNE_models(input_all_models, all_nodes):
    final_model = dict()
    for tmp_model in input_all_models:
        for node in all_nodes:
            if node in final_model:
                if node in tmp_model.wv.index2word:
                    final_model[node] = np.concatenate((final_model[node], tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]), axis=0)
                else:
                    final_model[node] = np.concatenate((final_model[node], np.zeros([args.dimensions])), axis=0)
            else:
                if node in tmp_model.wv.index2word:
                    final_model[node] = tmp_model.wv.syn0[tmp_model.wv.index2word.index(node)]
                else:
                    final_model[node] = np.zeros([args.dimensions])
    return final_model


def Evaluate_PMNE_methods(merged_networks, network_one, network_two, PMNE_3_network, node_label):
    # we need to write codes to implement the co-analysis method of PMNE
    # print('Start to analyze the PMNE method')
    all_network = merged_networks

    G = Random_walk.RWGraph(get_G_from_edges(all_network), args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model_one = train_deepwalk_embedding(walks)
    method_one_performance = get_auc(model_one, node_label)
    # print('Performance of PMNE method one:', method_one_performance)

    G = Random_walk.RWGraph(get_G_from_edges(network_one), args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model_two_up = train_deepwalk_embedding(walks)

    G = Random_walk.RWGraph(get_G_from_edges(network_two), args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model_two_down = train_deepwalk_embedding(walks)
    method_two_performance = get_PMNE_two_auc(model_two_up, model_two_down, node_label)
    # print('Performance of PMNE method two:', method_two_performance)

    tmp_graphs = list()
    for et in ['1', '2']:
        tmp_G = get_G_from_edges(PMNE_3_network[et])
        tmp_graphs.append(tmp_G)
    MK_G = Node2Vec_LayerSelect.Graph(tmp_graphs, args.p, args.q, 0.5)
    MK_G.preprocess_transition_probs()
    MK_walks = MK_G.simulate_walks(args.num_walks, args.walk_length)
    model_three = train_deepwalk_embedding(MK_walks)
    method_three_performance = get_auc(model_three, node_label)
    # print('Performance of PMNE method three:', method_three_performance)
    return method_one_performance, method_two_performance, method_three_performance


args = parse_args()
file_name = args.input
edge_type = ('1', '2')
num_of_groups = 5
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)
node_label = pd.DataFrame()

for threshold in [0.5, 0.8, 0.99]:
    all_node2vec_performance = list()
    all_deepwalk_performance = list()
    all_LINE_performance = list()
    all_PMNE_one_performance = list()
    all_PMNE_two_performance = list()
    all_PMNE_three_performance = list()
    all_convi_structure_performance = list()
    all_convi_property_performance = list()
    for nog in range(num_of_groups):
        if 'ee' in file_name.split('.edges')[0]:
            node_label = pd.read_csv('./threshold_data/eclipse_{}.csv'.format(threshold))
            LINE_path = './data/LINE/ee_'
        else:
            node_label = pd.read_csv('./threshold_data/gnome_{}.csv'.format(threshold))
            LINE_path = './data/LINE/ge_'

    # baseline node2vec deepwalk line PMNE================================================================================
        node2vec_auc = 0
        Deepwalk_auc = 0
        LINE_auc = 0
        for et in edge_type:
            node2vec_G = Random_walk.RWGraph(get_G_from_edges(edge_data_by_type[et]), args.directed, 2, 0.5)
            node2vec_G.preprocess_transition_probs()
            node2vec_walks = node2vec_G.simulate_walks(10, 10)
            node2vec_model = train_deepwalk_embedding(node2vec_walks)
            tmp_node2vec_auc = get_auc(node2vec_model, node_label)

            Deepwalk_G = Random_walk.RWGraph(get_G_from_edges(edge_data_by_type[et]), args.directed, 1, 1)
            Deepwalk_G.preprocess_transition_probs()
            Deepwalk_walks = Deepwalk_G.simulate_walks(args.num_walks, 10)
            Deepwalk_model = train_deepwalk_embedding(Deepwalk_walks)
            tmp_Deepwalk_auc = get_auc(Deepwalk_model, node_label)

            LINE_file = LINE_path + str(et) + '.edges'
            LINE_model = train_LINE_model(LINE_file)
            tmp_LINE_auc = get_dict_auc(LINE_model, node_label)

            node2vec_auc += tmp_node2vec_auc
            Deepwalk_auc += tmp_Deepwalk_auc
            LINE_auc += tmp_LINE_auc

        node2vec_auc = round(node2vec_auc/2.0, 2)
        Deepwalk_auc = round(Deepwalk_auc/2.0, 2)
        LINE_auc = round(LINE_auc/2.0, 2)
        # print('threshold:{}, node2vec_auc:{}'.format(threshold, node2vec_auc))
        # print('threshold:{}, Deepwalk_auc:{}'.format(threshold, Deepwalk_auc))
        # print('threshold:{}, LINE_auc:{}'.format(threshold, LINE_auc))
        all_node2vec_performance.append(node2vec_auc)
        all_deepwalk_performance.append(Deepwalk_auc)
        all_LINE_performance.append(LINE_auc)
        # PMNE
        edge_data_by_type_list = edge_data_by_type['1'] + edge_data_by_type['2']
        edge_data_by_type_df = pd.DataFrame(edge_data_by_type_list, columns=['from', 'to', 'weight'])
        edge_data_by_type_df['weight'] = edge_data_by_type_df['weight'].apply(int)
        merged_networks = edge_data_by_type_df.groupby(['from', 'to']).sum().reset_index()
        PMNE_3_network = edge_data_by_type
        # PMNE_network data 1:merged networks, 2:edge_data_by_type['1'],edge_data_by_type['2'], 3:PMNE_3_network
        PMNE_1_auc, PMNE_2_auc, PMNE_3_auc = Evaluate_PMNE_methods(merged_networks, edge_data_by_type['1'], edge_data_by_type['2'], PMNE_3_network, node_label) 
        all_PMNE_one_performance.append(PMNE_1_auc)
        all_PMNE_two_performance.append(PMNE_2_auc)
        all_PMNE_three_performance.append(PMNE_3_auc)

    # convi graph structure=====================================================================================
        convi_data = edge_data_by_type['2']
        weight_structure = [0.5, 0, 0, 0.5] # need to try
        cp = 0.5 # need to try
        convi_proper_G = Convince_Graph.ConvGraph(get_G_from_edges(edge_data_by_type['1']),\
                                                get_G_from_edges(convi_data),\
                                                args.directed, cp, weight_structure)
        convi_proper_G.preprocess_transition_probs(weight_structure)
        convi_MNE_walks = convi_proper_G.simulate_walks(10, 10)
        convi_MNE_model = train_deepwalk_embedding(convi_MNE_walks)
        tmp_convi_MNE_score = get_auc(convi_MNE_model, node_label)
        all_convi_structure_performance.append(tmp_convi_MNE_score)
        # print('threshold:{}, convi_graph_structure:{}'.format(threshold, tmp_convi_MNE_score))

        weight_property = [0, 0, 0.3, 0.4, 0.3]
        cp = 0.9
        convi_data = pd.DataFrame(convi_data, columns=['From', 'To', 'Weight'])
        convi_data['Weight'].apply(int)
        convi_proper_G = Convince_Multi_Graph.ConvMultiGraph(get_G_from_edges(edge_data_by_type['1']),\
                                                            get_G_from_edges(convi_data),\
                                                            edge_data_by_type['1'],\
                                                            convi_data, args.directed, cp, weight_property)
        convi_proper_G.preprocess_transition_probs(weight_property)
        convi_MNE_proper_walks = convi_proper_G.simulate_walks(10, 10)
        convi_MNE_proper_model = train_deepwalk_embedding(convi_MNE_proper_walks)
        tmp_convi_MNE_proper_score = get_auc(convi_MNE_proper_model, node_label)
        all_convi_property_performance.append(tmp_convi_MNE_proper_score)
        # print('threshold:{}, convi_graph_proper:{}'.format(threshold, tmp_convi_MNE_proper_score))

    print('threshold:{}'.format(threshold))
    print('all_node2vec_performance:{}, mean:{}, std:{}'.format\
    (all_node2vec_performance, np.mean(all_node2vec_performance), np.std(all_node2vec_performance)))
    print('all_deepwalk_performance:{}, mean:{}, std:{}'.format\
    (all_deepwalk_performance, np.mean(all_deepwalk_performance), np.std(all_deepwalk_performance)))
    print('all_LINE_performance:{}, mean:{}, std:{}'.format\
    (all_LINE_performance, np.mean(all_LINE_performance), np.std(all_LINE_performance)))
    print('all_PMNE_one_performance:{}, mean:{}, std:{}'.format\
    (all_PMNE_one_performance, np.mean(all_PMNE_one_performance), np.std(all_PMNE_one_performance)))
    print('all_PMNE_two_performance:{}, mean:{}, std:{}'.format\
    (all_PMNE_two_performance, np.mean(all_PMNE_two_performance), np.std(all_PMNE_two_performance)))
    print('all_PMNE_three_performance:{}, mean:{}, std:{}'.format\
    (all_PMNE_three_performance, np.mean(all_PMNE_three_performance), np.std(all_PMNE_three_performance)))
    print('all_convi_structure_performance:{}, mean:{}, std:{}'.format\
    (all_convi_structure_performance, np.mean(all_convi_structure_performance), np.std(all_convi_structure_performance)))
    print('all_convi_property_performance:{}, mean:{}, std:{}'.format\
    (all_convi_property_performance, np.mean(all_convi_property_performance), np.std(all_convi_property_performance)))

    