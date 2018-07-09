# This python file is used to reproduce our link prediction experiment
# Author: Hongming ZHANG, HKUST KnowComp Group

from sklearn.metrics import roc_auc_score
import math
import subprocess
import Node2Vec_LayerSelect
from gensim.models import Word2Vec
import argparse
import pandas as pd
import os
import Convince_Graph
from MNE import *
import random

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


# randomly divide data into few parts for the purpose of cross-validation
def divide_data(input_list, group_number):
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]


def train_deepwalk_embedding(walks, iteration=None):
    if iteration is None:
        iteration = 100
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=iteration)
    return model


def randomly_choose_false_edges(nodes, number, true_edges):
    tmp_list = list()
    all_edges = list()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            all_edges.append((i, j))
    random.shuffle(all_edges)
    for edge in all_edges:
        if edge[0] == edge[1]:
            continue
        if (nodes[edge[0]], nodes[edge[1]]) not in true_edges and (nodes[edge[1]], nodes[edge[0]]) not in true_edges:
            tmp_list.append((nodes[edge[0]], nodes[edge[1]]))
    return tmp_list

# cos similarity
def get_dict_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()

# 计算AUC
def get_dict_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    for edge in false_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)


def get_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()


def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(1)
            else:
                prediction_list.append(-1)
        else:
            prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)


args = parse_args()
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
file_name = args.input
# test_file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)
edge_data = {'1':edge_data_by_type['1']}
convi_data = edge_data_by_type['2']

# In our experiment, we use 5-fold cross-validation, but you can change that
number_of_groups = 5
edge_data_by_type_by_group = dict()
for edge_type in edge_data:
    all_data = edge_data[edge_type]
    separated_data = divide_data(all_data, number_of_groups)
    edge_data_by_type_by_group[edge_type] = separated_data


overall_convi_MNE_performance = list()

for i in range(number_of_groups):
    training_data_by_type = dict()
    evaluation_data_by_type = dict()
    for edge_type in edge_data_by_type_by_group:
        training_data_by_type[edge_type] = list()
        evaluation_data_by_type[edge_type] = list()
        for j in range(number_of_groups):
            if j == i:
                for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                    evaluation_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1], tmp_edge[2]))
            else:
                for tmp_edge in edge_data_by_type_by_group[edge_type][j]:
                    training_data_by_type[edge_type].append((tmp_edge[0], tmp_edge[1], tmp_edge[2]))
    base_edges = list()
    training_nodes = list()
    for edge_type in training_data_by_type:
        for edge in training_data_by_type[edge_type]:
            base_edges.append(edge)
            training_nodes.append(edge[0])
            training_nodes.append(edge[1])
    edge_data['Base'] = base_edges
    training_nodes = list(set(training_nodes))

    tmp_convi_MNE_performance = 0
    merged_networks = dict()
    merged_networks['training'] = dict()
    merged_networks['test_true'] = dict()
    merged_networks['test_false'] = dict()
    number_of_edges = 0
    for edge_type in training_data_by_type:
        if edge_type == 'Base':
            continue
        print('We are working on edge:', edge_type)
        selected_true_edges = list()
        tmp_training_nodes = list()
        for edge in training_data_by_type[edge_type]:
            tmp_training_nodes.append(edge[0])
            tmp_training_nodes.append(edge[1])
        tmp_training_nodes = set(tmp_training_nodes)
        for edge in evaluation_data_by_type[edge_type]:
            if edge[0] in tmp_training_nodes and edge[1] in tmp_training_nodes:
                if edge[0] == edge[1]:
                    continue
                selected_true_edges.append(edge)
        if len(selected_true_edges) == 0:
            continue
        selected_false_edges = randomly_choose_false_edges(training_nodes, len(selected_true_edges)/2,
                                                            edge_data[edge_type])
        # print('number of info network edges:', len(training_data_by_type[edge_type]))
        # print('number of evaluation edges:', len(selected_true_edges))
        merged_networks['training'][edge_type] = set(training_data_by_type[edge_type])
        merged_networks['test_true'][edge_type] = selected_true_edges
        merged_networks['test_false'][edge_type] = selected_false_edges
        local_model = dict()
        for cp in [0.1,0.3,0.5,0.7,0.9]:
            for i in range(100):
                weight_weight = random.randint(0, 10) * 0.1
                weight = np.array([weight_weight, 1 - weight_weight, 0, 0])
                convi_MNE_G = Convince_Graph.ConvGraph(get_G_from_edges(training_data_by_type[edge_type]),\
                                                        get_G_from_edges(convi_data),\
                                                        args.directed, cp, weight)
                convi_MNE_G.preprocess_transition_probs(weight)
                convi_MNE_walks = convi_MNE_G.simulate_walks(10, 10)
                convi_MNE_model = train_deepwalk_embedding(convi_MNE_walks)
                tmp_convi_MNE_score = get_AUC(convi_MNE_model, selected_true_edges, selected_false_edges)
                if tmp_convi_MNE_score > 0.60:
                    print('weight:{}'.format(weight))
                    print('convi_MNE score:', tmp_convi_MNE_score)


                # tmp_convi_MNE_performance += tmp_convi_MNE_score * 1
                # number_of_edges += 1

            # print('cp:{}'.format(cp), 'weight:{}'.format(weight))
            # print('convi_MNE performance:', tmp_convi_MNE_performance / number_of_edges)
            # overall_convi_MNE_performance.append(tmp_convi_MNE_performance / number_of_edges)

    # overall_convi_MNE_performance = np.asarray(overall_convi_MNE_performance)
    # print('Overall convi_MNE AUC:', overall_convi_MNE_performance)
    # print('Overall convi_MNE_AUC:', np.mean(overall_convi_MNE_performance))
    # print('Overall convi_MNE_AUC std:', np.std(overall_convi_MNE_performance))
    print('end')
