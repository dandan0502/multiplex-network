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
import Convince_Multi_Graph
from MNE import *
import random
import csv

def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

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
# def divide_data(input_list, group_number):
#     local_division = len(input_list) / float(group_number)
#     random.shuffle(input_list)
#     return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in range(group_number)]


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
# def get_dict_AUC(model, true_edges, false_edges):
#     true_list = list()
#     prediction_list = list()
#     for edge in true_edges:
#         tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
#         true_list.append(1)
#         # prediction_list.append(tmp_score)
#         # for the unseen pair, we randomly give a prediction
#         if tmp_score > 2:
#             if tmp_score > 2.5:
#                 prediction_list.append(1)
#             else:
#                 prediction_list.append(-1)
#         else:
#             prediction_list.append(tmp_score)
#     for edge in false_edges:
#         tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
#         true_list.append(0)
#         # prediction_list.append(tmp_score)
#         # for the unseen pair, we randomly give a prediction
#         if tmp_score > 2:
#             if tmp_score > 2.5:
#                 prediction_list.append(1)
#             else:
#                 prediction_list.append(-1)
#         else:
#             prediction_list.append(tmp_score)
#     y_true = np.array(true_list)
#     y_scores = np.array(prediction_list)
#     return roc_auc_score(y_true, y_scores)


# def get_neighbourhood_score(local_model, node1, node2):
#     try:
#         vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
#         vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
#         return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
#     except:
#         return 2+random.random()


# def get_AUC(model, true_edges, false_edges):
#     true_list = list()
#     prediction_list = list()
#     for edge in true_edges:
#         tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
#         true_list.append(1)
#         # prediction_list.append(tmp_score)
#         # for the unseen pair, we randomly give a prediction
#         if tmp_score > 2:
#             if tmp_score > 2.5:
#                 prediction_list.append(1)
#             else:
#                 prediction_list.append(-1)
#         else:
#             prediction_list.append(tmp_score)

#     for edge in false_edges:
#         tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
#         true_list.append(0)
#         # prediction_list.append(tmp_score)
#         # for the unseen pair, we randomly give a prediction
#         if tmp_score > 2:
#             if tmp_score > 2.5:
#                 prediction_list.append(1)
#             else:
#                 prediction_list.append(-1)
#         else:
#             prediction_list.append(tmp_score)
#     y_true = np.array(true_list)
#     y_scores = np.array(prediction_list)
#     return roc_auc_score(y_true, y_scores)


args = parse_args()
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
file_name = args.input
output_name = file_name.split('.edges')[0].split('data/')[1]
# test_file_name = 'data/Vickers-Chan-7thGraders_multiplex.edges'
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)
edge_data_by_type['1'] = pd.DataFrame(edge_data_by_type['1'], columns=['From', 'To', 'Weight'])
edge_data_by_type_df = edge_data_by_type['1']
edge_data_by_type['1']['Weight'].apply(int)
edge_data_by_type['1'] = edge_data_by_type['1'].values.tolist()
edge_data = {'1':edge_data_by_type['1']}

convi_data = edge_data_by_type['2']
convi_data = pd.DataFrame(convi_data, columns=['From', 'To', 'Weight'])
convi_data['Weight'].apply(int)

# In our experiment, we use 5-fold cross-validation, but you can change that
number_of_groups = 2
edge_data_by_type_by_group = dict()
all_data = edge_data['1']

# 找到tossing path 里的助攻与fixer的边概率
path = "../assists_dev_MNE/"
bugid_tosser_fixer = pd.read_csv(path + file_name.split('.')[0] + "_bugid_tossers_fixers.csv")
bugid_fixer_tosser_dict = dict(zip(bugid_tosser_fixer['bugid'] + '--' + bugid_tosser_fixer['fixers'], bugid_tosser_fixer['tossers']))
fixer_tosser_dict = dict(zip(bugid_tosser_fixer['fixers'], bugid_tosser_fixer['tossers']))
for fixer in fixer_tosser_dict:
    tossers = fixer_tosser_dict[fixer].split('++')

    for cp in [0.9, 0.8, 0.7, 0.6]:
        weight = np.array([0.1, 0.2, 0.3, 0.4])
        # overall_convi_MNE_performance = list()
        for i in range(number_of_groups):
            convi_struct_G = Convince_Graph.ConvGraph(get_G_from_edges(edge_data_by_type['1']),\
                                                    get_G_from_edges(convi_data),\
                                                    args.directed, cp, weight)
            # print("cross validation :{}",format(i))
            convi_struct_G.preprocess_transition_probs(weight)
            convi_MNE_walks = convi_struct_G.simulate_walks(10, 10)
            convi_MNE_model = train_deepwalk_embedding(convi_MNE_walks)

            # tmp_convi_MNE_score = get_AUC(convi_MNE_model, selected_true_edges, selected_false_edges)
            # print('weight:{}'.format(weight))

            # tmp_convi_MNE_performance += tmp_convi_MNE_score * 1
            # number_of_edges += 1
            
            # print('convi_MNE performance:', tmp_convi_MNE_performance / number_of_edges)
            # overall_convi_MNE_performance.append(tmp_convi_MNE_performance / number_of_edges)

        # print('cp:{}'.format(cp), 'weight:{}'.format(weight))
        # overall_convi_MNE_performance = np.asarray(overall_convi_MNE_performance)
        # print('Overall convi_MNE AUC:', overall_convi_MNE_performance)
        # print('Overall convi_MNE_AUC:', np.mean(overall_convi_MNE_performance))
        # print('Overall convi_MNE_AUC std:', np.std(overall_convi_MNE_performance))
        # with open('../result/baseline_node_link/link_prediction/{}_structure.csv'.format(output_name), 'a') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([cp, weight, overall_convi_MNE_performance, np.mean(overall_convi_MNE_performance), np.std(overall_convi_MNE_performance)])
    print('end')
                        
# ======================================property============================================================
# for fixer in fixer_tosser_dict:
#     tossers = fixer_tosser_dict[fixer].split('++')
#     for cp in [0.9, 0.8, 0.7]:
#         weight_property = np.array([0.1, 0.1, 0.3, 0.2, 0.3])
#         overall_convi_MNE_performance = list()
#         for i in range(number_of_groups):
            
#             convi_proper_G = Convince_Multi_Graph.ConvMultiGraph(get_G_from_edges(edge_data_by_type['1']),\
#                                                         get_G_from_edges(convi_data),\
#                                                         edge_data_by_type_df,\
#                                                         convi_data, args.directed, cp, weight_property)
#             # print("cross validation :{}",format(i))
#             convi_proper_G.preprocess_transition_probs(weight_property)
#             convi_MNE_walks = convi_proper_G.simulate_walks(10, 10)
#             convi_MNE_model = train_deepwalk_embedding(convi_MNE_walks)
#         #     tmp_convi_MNE_score = get_AUC(convi_MNE_model, selected_true_edges, selected_false_edges)

#         #     tmp_convi_MNE_performance += tmp_convi_MNE_score * 1
#         #     number_of_edges += 1

#         #     overall_convi_MNE_performance.append(tmp_convi_MNE_performance / number_of_edges)
        
#         # print('cp:{}'.format(cp), 'weight:{}'.format(weight_property))
#         # # print('convi_MNE performance:', tmp_convi_MNE_performance / number_of_edges)
#         # overall_convi_MNE_performance = np.asarray(overall_convi_MNE_performance)
#         # # print('Overall convi_MNE AUC:', overall_convi_MNE_performance)
#         # print('Overall convi_MNE_AUC:', np.mean(overall_convi_MNE_performance))
#         # # print('Overall convi_MNE_AUC std:', np.std(overall_convi_MNE_performance))

#         # with open('../result/baseline_node_link/link_prediction/{}_property.csv'.format(output_name), 'a') as csvfile:
#         #     writer = csv.writer(csvfile)
#         #     writer.writerow([cp, weight_property, overall_convi_MNE_performance, np.mean(overall_convi_MNE_performance), np.std(overall_convi_MNE_performance)])
#         print('end')
