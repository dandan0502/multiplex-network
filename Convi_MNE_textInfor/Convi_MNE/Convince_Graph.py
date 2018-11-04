from __future__ import division
import numpy as np
import networkx as nx
import random
import math
from get_score import *

class ConvGraph():
    def __init__(self, nx_G, Conv_G, is_directed, conv_p, weight):
        self.G = nx_G
        self.Conv_G = Conv_G
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


    def preprocess_transition_probs(self, weight):
        # Preprocessing of transition probabilities for guiding the random walks.
        G = self.G
        Conv_G = self.Conv_G
        Conv_p = self.p
        Inconv_p = 1 - Conv_p
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_prob = []
            for nbr in sorted(G.neighbors(node)):
                Connected, CN, JA, AA = get_structure_score(G, Conv_G, node, nbr)
                unnormalized_prob.append((nbr, count_structure_score(Connected, CN, JA, AA, weight)))
            # print(unnormalized_prob)
            unnormalized_probs = [weight * Conv_p \
                                    if nbr in Conv_G.neighbors(node) \
                                    else weight * Inconv_p for nbr, weight in unnormalized_prob]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const if norm_const != 0 else 0 for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        self.alias_nodes = alias_nodes

        return


    # def preprocess_transition_probs(self, weight):
    #     # Preprocessing of transition probabilities for guiding the random walks.
    #     G = self.G
    #     Conv_G = self.Conv_G
    #     Conv_p = self.p
    #     Inconv_p = 1 - Conv_p
    #     is_directed = self.is_directed

    #     alias_nodes = {}
    #     for node in G.nodes():
    #         unnormalized_prob = []
    #         for nbr in sorted(G.neighbors(node)):
    #             degree = get_property_score(G, node, nbr)
    #             unnormalized_prob.append((nbr, count_property_score(degree, weight)))
    #         # print(unnormalized_prob)
    #         unnormalized_probs = [weight * Conv_p \
    #                                 if nbr in Conv_G.neighbors(node) \
    #                                 else weight * Inconv_p for nbr, weight in unnormalized_prob]
    #         norm_const = sum(unnormalized_probs)
    #         normalized_probs = [float(u_prob) / norm_const if norm_const != 0 else 0 for u_prob in unnormalized_probs]
    #         alias_nodes[node] = alias_setup(normalized_probs)

    #     self.alias_nodes = alias_nodes

    #     return


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
