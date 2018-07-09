# -*- coding: utf-8-*-
import networkx as nx
import pandas as pd
import random

def load_single_layer_data(path_file):
	'''
	the aaaa is included in graph, but the weight of the edges linked to aaaa is 0
	'''
	df = pd.read_csv(path_file, sep=' ', header=None)
	return df

def str2index(df):
	name = sorted(list(set(df.From) | set(df.To)))
	# name.remove('aaaa')
	index = range(len(name))
	name_index = pd.DataFrame()
	name_index['From'] = name
	name_index['To'] = name
	name_index['index'] = index
	df = df.merge(name_index, on='From', how='left')
	df = df.drop(['To_y'], axis=1)
	df.columns=['From', 'To', 'weight', 'index']
	df = df.merge(name_index, on='To', how='left')
	df = df.drop(['From_x', 'To', 'From_y'], axis=1)
	df.columns=['weight', 'From', 'To']
	df = df[['From', 'To', 'weight']]
	return df
	
def build_g(df, isDi, isWe):
	triad = list(zip(*[df[c].values.tolist() for c in df]))
	if isDi:
		G = nx.DiGraph()
		G = G.to_directed()
	else:
		G = nx.Graph()
	if isWe:
		G.add_weighted_edges_from(triad)
	else:
		G.add_edges_from(triad)
	return G

def bin_op(graph, df):
	# positive example of data
	large_scc = max(nx.strongly_connected_components(graph), key=len)
	graph_nodes = graph.nodes()
	other_nodes = set(graph_nodes) - large_scc
	graph.remove_nodes_from(list(other_nodes))
	pos_df = pd.DataFrame(list(graph.edges()), columns=['From', 'To'])
	# negative example of data
	neg_dict = {} # avoid duplicate value
	# random.seed(1)
	while(len(neg_dict) <= len(graph.edges())): # the length of negative examples is the same as positive examples
		from_index = random.randint(0, len(graph.nodes) - 1)
		to_index = random.randint(0, len(graph.nodes) - 1)
		tmp_index = str(from_index) + '-' + str(to_index)
		graph_edges = [str(i[0]) + '-' + str(i[1]) for i in graph.edges()]
		if (tmp_index not in graph_edges) and (tmp_index not in neg_dict)\
			 and (from_index not in other_nodes) and (to_index not in other_nodes):
			neg_dict[tmp_index] = 1
	# convert negative example dict to dataframe
	neg_df = pd.DataFrame(neg_dict.items(), columns=['From', 'weight'])
	neg_df.insert(1, 'To', neg_df['From'])
	neg_df['From'] = neg_df['From'].map(lambda x:str(x).split('-')[0])
	neg_df['To'] = neg_df['To'].map(lambda x:str(x).split('-')[1])
	neg_df.drop(['weight'], axis=1, inplace=True)
	data_df = pd.concat([pos_df, neg_df])

	# bin_op
	node_dict = {}
	for row in df.iterrows():
		node_dict[str(int(row[1][0]))] = pd.DataFrame(row[1][1:])
		node_dict[str(int(row[1][0]))].columns = ['a']
	# edge vector matrix dict,{edge1:vec1,edge2:vec2,...}
	edge_matrix = {}
	for row in data_df.iterrows():
		edge = str(row[1][0]) + '-' + str(row[1][1])
		# bin_result = ((node_dict[str(row[1][0])] + node_dict[str(row[1][1])]) / 2).values.tolist()
		bin_result = pd.concat([node_dict[str(row[1][0])], node_dict[str(row[1][1])]], axis=1).values.tolist()
		edge_matrix[edge] = [value for _list in bin_result for value in _list]
	# convert edge vector matrix dict to dataframe
	L = []
	for key in edge_matrix:
		tmplist = [key] + edge_matrix[key]
		L.append(tmplist)
	edge_matrix_df = pd.DataFrame(L, columns=['From'] + \
									['feature_{}'.format(i) for i in range(1, df.shape[1] * 2 - 1)])
	edge_matrix_df.insert(1, 'To', edge_matrix_df['From'])
	edge_matrix_df['From'] = edge_matrix_df['From'].map(lambda x:str(x).split('-')[0])
	edge_matrix_df['To'] = edge_matrix_df['To'].map(lambda x:str(x).split('-')[1])
	edge_matrix_df['flag'] = 1
	edge_matrix_df['flag'][int(len(edge_matrix_df)/2):] = 0
	edge_matrix_df.drop(['From', 'To'], axis=1, inplace=True)
	# print(edge_matrix_df.head())
	# edge_matrix_df.to_csv('./result/load_data/' + platform +'_data.csv', index=None)
	return edge_matrix_df
	
def format_libsvm(edge_matrix):
	edge_matrix.insert(0, 'label', edge_matrix['flag'])
	edge_matrix.drop(['flag'], axis=1, inplace=True)
	tmp = edge_matrix.values.tolist()
	for row in tmp:
		for i in range(1, len(row)):
			row[i] = str(i - 1) + ':' + str(row[i])
	return tmp