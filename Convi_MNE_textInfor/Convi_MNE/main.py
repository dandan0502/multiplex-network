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


def get_tosser_fixer_prob(model, node1, node2):
    try:
        vector1 = model.wv.syn0[model.wv.index2word.index(node1)]
        vector2 = model.wv.syn0[model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 2+random.random()


def count_assists(l):
    tmp_ass = 0
    all_tossers_num = len(l['other_t']) + 1
    for p in l['other_t']:
        if float(l['fixers']) <= p:
            tmp_ass += 1
    l['count'] = tmp_ass/all_tossers_num
    return l


args = parse_args()
file_name = args.input
output_name = file_name.split('.edges')[0].split('data/')[1]
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
if file_name.split('.')[0].split('/')[1] == 'ee':
    file_name_pre = 'eclipse'
else:
    file_name_pre = 'gnome'
bugid_tosser_fixer = pd.read_csv(path + file_name_pre + "_bugid_tossers_fixers.csv")
bugid_tosser_fixer['bugid'] = bugid_tosser_fixer['bugid'].astype("str")
bugid_tosser_fixer['fixers'] = bugid_tosser_fixer['fixers'].astype("str")
bugid_fixer_tosser_dict = dict(zip(bugid_tosser_fixer['bugid'] + '--' + bugid_tosser_fixer['fixers'], bugid_tosser_fixer['tossers']))
# fixer_tosser_dict = dict(zip(bugid_tosser_fixer['fixers'], bugid_tosser_fixer['tossers']))
for i in range(number_of_groups):
    for cp in [0.9, 0.8]:
        weight = np.array([0.1, 0.2, 0.3, 0.4])
        fixer_lastTosser_tosser_df = pd.DataFrame()
        fixer_lastTosser_tosser_df['fixers'] = list(map(lambda x:x.split('--')[1], bugid_fixer_tosser_dict))
        last_tosser = []
        other_tossers = []
        convi_struct_G = Convince_Graph.ConvGraph(get_G_from_edges(edge_data_by_type['1']),\
                                                get_G_from_edges(convi_data),\
                                                args.directed, cp, weight)
        convi_struct_G.preprocess_transition_probs(weight)
        convi_MNE_walks = convi_struct_G.simulate_walks(10, 10)
        convi_MNE_model = train_deepwalk_embedding(convi_MNE_walks)

        # 固定fixer找到路径上的所有tosser与fixer的边概率
        for bugid_fixer in bugid_fixer_tosser_dict:
            tossers = bugid_fixer_tosser_dict[bugid_fixer].split('++')
            prob_last = get_tosser_fixer_prob(convi_MNE_model, tossers[-1], bugid_fixer.split('--')[1])
            last_tosser.append(prob_last)
            tmp_tosser = []
            for tosser in tossers[-1]:
                prob_tosser_fixer = get_tosser_fixer_prob(convi_MNE_model, tosser, bugid_fixer.split('--')[1])
                tmp_tosser.append(prob_tosser_fixer)
            other_tossers.append(tmp_tosser)
        fixer_lastTosser_tosser_df['last_t'] = pd.Series(last_tosser)
        fixer_lastTosser_tosser_df['other_t'] = pd.Series(other_tossers)
        fixer_lastTosser_tosser_df = fixer_lastTosser_tosser_df.apply(count_assists, axis=1)
        print(len(fixer_lastTosser_tosser_df), len(fixer_lastTosser_tosser_df[fixer_lastTosser_tosser_df['count']>0]))
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
