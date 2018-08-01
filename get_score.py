from __future__ import division
import numpy as np
import pandas as pd
import math
import scipy
from collections import Counter
import networkx as nx

def get_common_neighbor_score(networks, target_A, target_B):
    common_neighbor_counter = 0
    A_neighbors = list()
    B_neighbors = list()
    for edge in networks.edges():
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            common_neighbor_counter += 1
    return common_neighbor_counter


def get_Jaccard_score(networks, target_A, target_B):
    A_neighbors = list()
    B_neighbors = list()
    for edge in networks.edges():
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    common_neighbor_counter = 0
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            common_neighbor_counter += 1
    if len(A_neighbors) == 0 and len(B_neighbors) == 0:
        Jaccard_score = 1
    else:
        Jaccard_score = common_neighbor_counter/(len(A_neighbors) + len(B_neighbors) - common_neighbor_counter)
    return Jaccard_score


def get_frequency_dict(networks):
    counting_dict = dict()
    for edge in networks.edges():
        if edge[0] not in counting_dict:
            counting_dict[edge[0]] = 0
        if edge[1] not in counting_dict:
            counting_dict[edge[1]] = 0
        counting_dict[edge[0]] += 1
        counting_dict[edge[1]] += 1
    return counting_dict


def get_AA_score(networks, target_A, target_B, frequency_dict):
    AA_score = 0
    A_neighbors = list()
    B_neighbors = list()
    for edge in networks.edges():
        if edge[0] == target_A:
            A_neighbors.append(edge[1])
        if edge[1] == target_A:
            A_neighbors.append(edge[0])
        if edge[0] == target_B:
            B_neighbors.append(edge[1])
        if edge[1] == target_B:
            B_neighbors.append(edge[0])
    for neighbor in A_neighbors:
        if neighbor in B_neighbors:
            if frequency_dict[neighbor] > 1:
                AA_score += 1/(math.log(frequency_dict[neighbor]))
    return AA_score


def get_structure_score(G, Conv_G, node, nbr):
    Connected = G[node][nbr]['weight']
    CN = get_common_neighbor_score(G, node, nbr)
    JA = get_Jaccard_score(G, node, nbr)
    frequency_dict = get_frequency_dict(G)
    AA = get_AA_score(G, node, nbr, frequency_dict)
    return Connected, CN, JA, AA


def count_structure_score(Connected, CN, JA, AA, weight):
    score = np.array([float(Connected), float(CN), float(JA), float(AA)])
    return sum(score * weight)


# multiplex network property
def get_oi(email, hist):
    email_oi = email.degree
    email_oi_d = {}
    for k, v in email_oi:
        email_oi_d[k] = v
    hist_oi = hist.degree
    hist_oi_d = {}
    for k, v in hist_oi:
        hist_oi_d[k] = v
    oi = {}
    for k, v in email_oi_d.items():
        oi[k] = hist_oi_d[k] + v
    # print(len(email.nodes()),len(hist_oi_d),len(email_oi_d),len(oi))
    return oi


def get_si(email, nbr):
    strength_df = email.groupby(['From'])[['Weight']].sum().reset_index()
    strength_df['Weight'] = scipy.stats.zscore(np.array(strength_df['Weight']))
    strength = round(float(strength_df[strength_df['From'] == nbr].iloc[0,1]), 4)
    return strength


def get_o_ij_w(email, hist):
    # email = pd.DataFrame(email, columns=['From', 'To', 'Weight'])
    # hist  = pd.DataFrame(hist,  columns=['From', 'To', 'Weight'])
    df = pd.concat([email, hist])
    df['Weight'] = df['Weight'].apply(int)
    weight_sum = df.groupby(['From', 'To'])[['Weight']].sum().reset_index()
    return weight_sum


def get_cross_entropy(email, hist, oi, nbr):
    # email ki
    email_oi = email.degree
    email_oi_d = {}
    for k, v in email_oi:
        email_oi_d[k] = v
    # hist ki
    hist_oi = hist.degree
    hist_oi_d = {}
    for k, v in hist_oi:
        hist_oi_d[k] = v
    # the distribution of every node on two layers
    email_dist = {}
    hist_dist  = {}
    Hi = {}
    for k,v in email_oi_d.items():
        if oi.get(k):
            email_dist[k] = round(float(email_oi_d[k])/oi[k],2)
    for k,v in hist_oi_d.items():
        if oi.get(k) and email_dist.get(k):
            hist_dist[k] = round(float(hist_oi_d[k])/oi[k],2)
            log_email = 0 if email_dist[k] == 0 else math.log(float(email_dist[k]))
            log_hist = 0 if hist_dist[k] == 0 else math.log(float(hist_dist[k]))
            Hi[k] = -round(email_dist[k]*log_email+hist_dist[k]*log_hist,3)
    return Hi[nbr]


def get_Ci1(email_layer, hist_layer, email_graph, hist_graph, nbr):
    # email = pd.DataFrame(email_layer, columns=['From', 'To', 'Weight'])
    # hist  = pd.DataFrame(hist_layer,  columns=['From', 'To', 'Weight'])
    email_layer = email_layer[['From', 'To']]
    hist_layer = hist_layer[['From', 'To']]
    # ki
    email_oi = email_graph.degree
    email_oi_d = {}
    for k, v in email_oi:
        email_oi_d[k] = v
    hist_oi = hist_graph.degree
    hist_oi_d = {}
    for k, v in hist_oi:
        hist_oi_d[k] = v
    # email layer
    # init
    triangle_2_email = dict.fromkeys(list(set(email_layer['From']) | set(email_layer['To'])), 0)
    # del triangle_2_email['aaaa']
    triad_1_email = dict.fromkeys(list(set(email_layer['From']) | set(email_layer['To'])), 0)
    # del triad_1_email['aaaa']
    # triangle_2
    for key,value in email_oi_d.items():
        key_row = email_layer[email_layer['From'] == key]
        for row in key_row.values.tolist():
            a = hist_layer[(hist_layer['From'] == row[1]) & (hist_layer['From']!=hist_layer['To'])]
            if not a.empty:
                for hist_row in a.values.tolist():
                    b = email_layer[(email_layer['From'] == hist_row[1]) & \
                                    (email_layer['From'] != email_layer['To']) & \
                                    (email_layer['To'] == row[0])]
                    if not b.empty:
                        triangle_2_email[key] += len(b)
        # triad_1
        for row in key_row.values.tolist():
            triad_1_email[key] = triad_1_email[key] + len(email_layer[(email_layer['To']==row[0]) & \
                                                    (email_layer['From']!=row[1])])
    # hist layer
    triangle_2_hist = dict.fromkeys(list(set(hist_layer['From']) | set(hist_layer['To'])),0)
    # del triangle_2_hist['aaaa']
    triad_1_hist = dict.fromkeys(list(set(hist_layer['From']) | set(hist_layer['To'])),0)
    # del triad_1_hist['aaaa']
    for key,value in hist_oi_d.items():
        key_row = hist_layer[hist_layer['From'] == key]
        for row in key_row.values.tolist():
            a = email_layer[(email_layer['From'] == row[1]) & (email_layer['From'] != email_layer['To'])]
            if not a.empty:
                for email_row in a.values.tolist():
                    b = hist_layer[(hist_layer['From'] == email_row[1]) & \
                                    (hist_layer['From'] != hist_layer['To']) & \
                                    (hist_layer['To'] == row[0])]
                    if not b.empty:
                        triangle_2_hist[key] += len(b)
        for row in key_row.values.tolist():
            triad_1_hist[key] = triad_1_hist[key] + len(hist_layer[(hist_layer['To'] == row[0]) & \
                                                    (hist_layer['From'] != row[1])])
    triangle_2 = dict(Counter(triangle_2_email) + Counter(triangle_2_hist))
    triad_1 = dict(Counter(triad_1_email) + Counter(triad_1_hist))
    Ci1 = {}
    for i in triangle_2.keys():
        Ci1[i] = round(triangle_2[i] / float(triad_1[i]), 2)
    if Ci1.get(nbr, 0) == nbr:
        return Ci1[nbr]
    else:
        return 0


def aggregated_network(email, hist):
    # email = pd.DataFrame(email, columns=['From', 'To', 'Weight'])
    # hist  = pd.DataFrame(hist,  columns=['From', 'To', 'Weight'])
    agg_network = pd.concat([email,hist])
    agg_network['Weight'] = agg_network['Weight'].apply(int)
    agg_network = agg_network.sort_values(['Weight'], ascending=True).drop_duplicates(['From','To'], keep='first')
    agg_network = agg_network[~(agg_network['From'] == agg_network['To'])]
    return agg_network


def get_lambdai(email_layer, hist_layer, email_graph, hist_graph, nbr):
    # aggregate network
    df = aggregated_network(email_layer, hist_layer)
    agg_G = nx.DiGraph()
    agg_G = agg_G.to_directed()
    triad = list(zip(*[df[c].values.tolist() for c in df]))
    agg_G.add_weighted_edges_from(triad)
    # agg_G.remove_node('aaaa')
    # total number of shortest paths on multiplex network
    agg_path   = nx.all_pairs_dijkstra_path(agg_G)
    email_path = nx.all_pairs_dijkstra_path(email_graph)
    hist_path  = nx.all_pairs_dijkstra_path(hist_graph)
    agg_dict, email_dict, hist_dict = {}, {}, {}
    email_edge_path_dict, hist_edge_path_dict, agg_edge_path_dict = {}, {}, {} # shortest paths of all nodes
    lambdai = {}
    for node, path in email_path:
        for i in path:
            email_edge_path_dict['++'.join(path[i])] = 1
    for node, path  in hist_path:
        for i in path:
            hist_edge_path_dict['++'.join(path[i])] = 1
    for node, path  in agg_path:
        for i in path:
            agg_edge_path_dict['++'.join(path[i])] = 1
    # count the shortest path of node i in agg_network
    bottom_df = pd.DataFrame(list(agg_edge_path_dict.items()), columns=['edge', 'count'])
    bottom_df['From'] = bottom_df['edge'].map(lambda x : x.split('++')[0])
    bottom_count = Counter(bottom_df['From'])
    for edge in email_edge_path_dict:
        if agg_edge_path_dict.get(edge, 'aaaa') != 'aaaa':
            agg_edge_path_dict.pop(edge)
    # count the shortest path of all nodes between two layers
    for edge in hist_edge_path_dict:
        if agg_edge_path_dict.get(edge, 'aaaa') != 'aaaa':
            agg_edge_path_dict.pop(edge)
    # count the shortest path of node i  between two layers
    top_df = pd.DataFrame(list(agg_edge_path_dict.items()), columns=['edge', 'count'])
    top_df['From'] = top_df['edge'].map(lambda x : x.split('++')[0])
    top_count = Counter(top_df['From'])
    # calculate the lambda i
    lambdai = {}
    for k in bottom_count:
        if top_count.get(k,'aaaa') != 'aaaa':
            lambdai[k] = round(top_count[k] / float(bottom_count[k]),2)
    if lambdai.get(nbr, 0) == nbr:
        return lambdai[nbr]
    else:
        return 0


def get_property_score(G, Conv_G, df, Conv_df, node, nbr):
    overlap_degree = get_oi(G, Conv_G)
    zscore_norm_overlap_degree = dict(zip(overlap_degree.keys(), scipy.stats.zscore(list(overlap_degree.values()))))
    nbr_strength = get_si(df, nbr)
    entropy_degree = get_cross_entropy(G, Conv_G, overlap_degree, nbr)
    first_coef = get_Ci1(df, Conv_df, G, Conv_G, nbr)
    inter_dependence = get_lambdai(df, Conv_df, G, Conv_G, nbr)
    # print(nbr_strength, entropy_degree, first_coef, inter_dependence)
    # print("property score finished")
    return zscore_norm_overlap_degree[nbr], nbr_strength, entropy_degree, first_coef, inter_dependence


def count_property_score(overlap_degree, strength, entropy_degree, first_coef, inter_dependence, weight):
    score = np.array([float(overlap_degree), float(strength), float(entropy_degree), float(first_coef), float(inter_dependence)])
    return sum(score * weight)