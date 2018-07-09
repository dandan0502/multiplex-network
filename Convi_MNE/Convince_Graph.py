from __future__ import division
import numpy as np
import networkx as nx
import random
import math

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

    def get_common_neighbor_score(self, networks, target_A, target_B):
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

    def get_Jaccard_score(self, networks, target_A, target_B):
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

    def get_frequency_dict(self, networks):
        counting_dict = dict()
        for edge in networks.edges():
            if edge[0] not in counting_dict:
                counting_dict[edge[0]] = 0
            if edge[1] not in counting_dict:
                counting_dict[edge[1]] = 0
            counting_dict[edge[0]] += 1
            counting_dict[edge[1]] += 1
        return counting_dict

    def get_AA_score(self, networks, target_A, target_B, frequency_dict):
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

    def get_score(self, G, Conv_G, node, nbr):
        # if nbr not in Conv_G.neighbors(node):
        #     Connected = 0.
        # else:
        Connected = G[node][nbr]['weight']
        CN = self.get_common_neighbor_score(G, node, nbr)
        JA = self.get_Jaccard_score(G, node, nbr)
        frequency_dict = self.get_frequency_dict(G)
        AA = self.get_AA_score(G, node, nbr, frequency_dict)
        return Connected, CN, JA, AA

    def count_score(self, Connected, CN, JA, AA, weight):
        score = np.array([float(Connected), float(CN), float(JA), float(AA)])
        # print(score)
        # weight = np.array([1, 1, 0, 0])
        return sum(score * weight)

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
                Connected, CN, JA, AA = self.get_score(G, Conv_G, node, nbr)
                unnormalized_prob.append((nbr, self.count_score(Connected, CN, JA, AA, weight)))
            # print(unnormalized_prob)
            unnormalized_probs = [weight * Conv_p \
                                    if nbr in Conv_G.neighbors(node) \
                                    else weight * Inconv_p for nbr, weight in unnormalized_prob]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const if norm_const != 0 else 0 for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)


        # alias_nodes = {}
        # for node in G.nodes():
        #     unnormalized_prob = []
        #     for nbr in sorted(G.neighbors(node)):
        #         unnormalized_prob.append((nbr, float(G[node][nbr]['weight'])))
        #     unnormalized_probs = [weight * Conv_p if nbr in Conv_G.neighbors(node) else weight * Inconv_p\
        #                             for nbr, weight in unnormalized_prob]
        #     norm_const = sum(unnormalized_probs)
        #     normalized_probs = [float(u_prob) / norm_const if norm_const != 0 else 0 for u_prob in unnormalized_probs]
        #     alias_nodes[node] = alias_setup(normalized_probs)

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
