from __future__ import division
import numpy as np
import networkx as nx
import random
import math
from get_score import *
import scipy

class ConvMultiGraph():
    def __init__(self, nx_G, Conv_G, df, Conv_df, is_directed, conv_p, weight):
        self.G = nx_G
        self.Conv_G = Conv_G
        self.df = df
        self.Conv_df = Conv_df
        self.is_directed = is_directed
        self.p = conv_p
        self.weight = weight


    def walk(self, walk_length, start_node):
        # Simulate a random walk starting from start node.
        G = self.G
        alias_nodes = self.alias_nodes

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
            else:
                break
        return walk


    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(walk_length=walk_length, start_node=node))

        return walks


    def convi_score(self):
        G = self.G
        Conv_G = self.Conv_G
        df = self.df
        df = pd.DataFrame(df, columns=['From', 'To', 'Weight'])
        df['Weight'] = np.array(df['Weight']).astype(int)
        Conv_df = self.Conv_df
        overlap_weight = get_o_ij_w(df, Conv_df)
        zscore_norm_overlap_weight = scipy.stats.zscore(np.array(overlap_weight['Weight']).astype(int))
        return zscore_norm_overlap_weight


    def preprocess_transition_probs(self, weight):
        # Preprocessing of transition probabilities for guiding the random walks.
        G = self.G
        Conv_G = self.Conv_G
        df = self.df
        df = pd.DataFrame(df, columns=['From', 'To', 'Weight'])
        df['Weight'] = np.array(df['Weight']).astype(int)
        Conv_df = self.Conv_df
        Conv_df = pd.DataFrame(Conv_df, columns=['From', 'To', 'Weight'])
        Conv_df['Weight'] = np.array(Conv_df['Weight']).astype(int)
        Conv_p = self.p
        Inconv_p = 1 - Conv_p
        is_directed = self.is_directed

        zscore_norm_overlap_weight = self.convi_score()
        overlap_degree, strength_df, entropy_degree, Ci1, lambdai = get_property_score(G, Conv_G, df, Conv_df, weight)

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_prob = []
            for nbr in sorted(G.neighbors(node)):
                try:
                    strength = round(float(strength_df[strength_df['From'] == nbr].iloc[0, 1]), 4)
                except :
                    strength = 0
                if Ci1.get(nbr, 0) == nbr:
                    first_coef = Ci1[nbr]
                else:
                    first_coef = 0
                if lambdai.get(nbr, 0) == nbr:
                    inter_dependence = lambdai[nbr]
                else:
                    inter_dependence = 0
                unnormalized_prob.append((nbr, count_property_score(overlap_degree[nbr], strength, entropy_degree[nbr], first_coef, inter_dependence, weight)))
            unnormalized_probs = [convi_weight * Conv_p \
                                    if nbr in Conv_G.neighbors(node) \
                                    else convi_weight * Inconv_p for nbr, convi_weight in unnormalized_prob]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const if norm_const != 0 else 0 for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        self.alias_nodes = alias_nodes

        return


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    # Draw sample from a non-uniform discrete distribution using alias sampling.
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
