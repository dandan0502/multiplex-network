# -*- coding: utf-8-*-  
import pandas as pd
import os
from collections import Counter
from collections import defaultdict
import networkx as nx
import math
import csv

def dict2csv(dict,file):
    with open(file,'wb') as f:
        w=csv.writer(f)
        w.writerows(dict.items())

def preprocess(files):
	for f in files:
		df = pd.read_csv('./raw_data/'+f, usecols=['From','To','When_UTC'])
		df = df.fillna('aaaa')
		df = df.drop_duplicates()
		df = df[~((df['From']=='aaaa') | (df['To']=='aaaa') | (df['When_UTC']=='aaaa'))]
		count = Counter(df.From+'>'+df.To)
		df = pd.DataFrame(count.items(), columns=['From', 'Weight'])
		df.insert(1,'To',df['From'])
		df['From'] = df['From'].map(lambda x:str(x).split('>')[0])
		df['To'] = df['To'].map(lambda x:str(x).split('>')[1])
		df = df[~(df['From'] == df['To'])]
		df.to_csv('./preprocess/'+f,index = None)

def func(row):
	if row['From'] == 'aaaa' or row['To'] == 'aaaa':
		row['Weight'] = 0
	return row

# node:eclipse 608 
#      gnome 892 (not include aaaa)
# link:eclipse 
	   # gnome
def common_node(platform, layer):
	for p in platform:
		historydata = pd.read_csv('./preprocess/'+p+'_'+layer[0]+'.csv', encoding='ISO-8859-1')
		emaildata = pd.read_csv('./preprocess/'+p+'_'+layer[1]+'.csv', encoding='ISO-8859-1')
		# 对两个平台分开处理
		if p == 'eclipse':
			historydata['From'] = historydata['From'].map(lambda x:str(x).split('@')[0])
			historydata['To'] = historydata['To'].map(lambda x:str(x).split('@')[0])
		else:
			historydata['From'] = historydata['From'].map(lambda x:str(x).replace('@',' ').replace('.',' '))
			historydata['To'] = historydata['To'].map(lambda x:str(x).replace('@',' ').replace('.',' '))

		# 构建multiplex network
		common_dev = (set(historydata['From']) | set(historydata['To'])) & \
					(set(emaildata['From']) | set(emaildata['To']))
		# print(len(common_dev))
		historydata2 = historydata[['From','To']].applymap(lambda x:x if x in common_dev else 'aaaa')
		historydata2['Weight'] = historydata['Weight']
		emaildata2 = emaildata[['From','To']].applymap(lambda x:x if x in common_dev else 'aaaa')
		emaildata2['Weight'] = emaildata['Weight']
		historydata2 = historydata2[~((historydata2['From']=='aaaa') & (historydata2['To']=='aaaa'))]
		emaildata2 = emaildata2[~((emaildata2['From']=='aaaa') & (emaildata2['To']=='aaaa'))]

		historydata2 = historydata2.groupby(['From','To']).sum().reset_index()
		emaildata2 = emaildata2.groupby(['From','To']).sum().reset_index()

		historydata2 = historydata2.apply(func,axis=1)
		emaildata2 = emaildata2.apply(func,axis=1)

		historydata2.to_csv('./common_node/'+p+'history.csv',index=None)
		emaildata2.to_csv('./common_node/'+p+'email.csv',index=None)

def load_data():
	eclipse_hist = pd.read_csv('./common_node/eclipsehistory.csv')
	eclipse_email = pd.read_csv('./common_node/eclipseemail.csv')
	gnome_hist = pd.read_csv('./common_node/gnomehistory.csv')
	gnome_email = pd.read_csv('./common_node/gnomeemail.csv')
	layer = {
		'eh':eclipse_hist,
		'ee':eclipse_email,
		'gh':gnome_hist,
		'ge':gnome_email,
	}
	return layer

# not include the node aaaa and the link linking with it
def build_graph(layer_data):
	graph = {}
	for l,df in layer_data.items():
		triad = list(zip(*[df[c].values.tolist() for c in df]))
		G = nx.DiGraph()
		G = G.to_directed()
		G.add_weighted_edges_from(triad)
		G.remove_node('aaaa')
		graph[l] = G
		# print(l,len(graph[l].edges()))
	return graph

def single_layer_property(layer_data):
	'''单层计算后合并求解'''

	_dict = {}
	# init
	for i in layer_data:
		_dict[i] = {'topK_degree_centrality':[],'topK_closeness_centrality':[],'topK_betweenness_centrality':[]}

	for l, df in layer_data.items():
		# 使用networkx计算网络的性质，包括4个网络的度、介数、中心性、连通性
		triad = list(zip(*[df[c].values.tolist() for c in df]))
		G = nx.DiGraph()
		G = G.to_directed()
		# G.add_weighted_edges_from(triad)
		G.add_nodes_from(list(set(df['From'])|set(df['To'])))
		G.remove_node('aaaa')
		edges = df[df['Weight']!=0][['From','To']]
		edges = zip(edges['From'],edges['To'])
		G.add_edges_from(edges)

		print(l,'largest scc',len(list(max(nx.strongly_connected_components(G), key=len))))
		print(l,'transitivity',nx.transitivity(G))

		# consistency of degree, closeness, betweenness at topK
		k = 100
		_dict[l]['topK_degree_centrality'] = sorted(nx.degree_centrality(G).iteritems(),key=lambda x:x[1],reverse=True)[:k]
		_dict[l]['topK_closeness_centrality'] = sorted(nx.closeness_centrality(G).iteritems(),key=lambda x:x[1],reverse=True)[:k]
		_dict[l]['topK_betweenness_centrality'] = sorted(nx.betweenness_centrality(G).iteritems(),key=lambda x:x[1],reverse=True)[:k]

	eclipse_topK_degree_commonP = len(set(zip(*_dict['eh']['topK_degree_centrality'])[0]) & \
										set(zip(*_dict['ee']['topK_degree_centrality'])[0]))
	eclipse_topK_closeness_commonP = len(set(zip(*_dict['eh']['topK_closeness_centrality'])[0]) & \
										set(zip(*_dict['ee']['topK_closeness_centrality'])[0]))
	eclipse_topK_betweenness_commonP = len(set(zip(*_dict['eh']['topK_betweenness_centrality'])[0]) & \
										set(zip(*_dict['ee']['topK_betweenness_centrality'])[0]))

	gnome_topK_degree_commonP = len(set(zip(*_dict['gh']['topK_degree_centrality'])[0]) & \
									set(zip(*_dict['ge']['topK_degree_centrality'])[0]))
	gnome_topK_closeness_commonP = len(set(zip(*_dict['gh']['topK_closeness_centrality'])[0]) & \
										set(zip(*_dict['ge']['topK_closeness_centrality'])[0]))
	gnome_topK_betweenness_commonP = len(set(zip(*_dict['gh']['topK_betweenness_centrality'])[0]) & 
											set(zip(*_dict['ge']['topK_betweenness_centrality'])[0]))
			
	print('eclipse_topK_degree_commonP',eclipse_topK_degree_commonP)
	print('eclipse_topK_closeness_commonP',eclipse_topK_closeness_commonP)
	print('eclipse_topK_betweenness_commonP',eclipse_topK_betweenness_commonP)
	print('gnome_topK_degree_commonP',gnome_topK_degree_commonP)
	print('gnome_topK_closeness_commonP',gnome_topK_closeness_commonP)
	print('gnome_topK_betweenness_commonP',gnome_topK_betweenness_commonP)

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

def get_o_ij_w(email, hist):
	df = pd.concat([email, hist])
	weight_sum = df.groupby(['From', 'To'])[['Weight']].sum().reset_index()
	return weight_sum	

def get_cross_entropy(email, hist, oi):
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
	return Hi

def get_pw(email, hist):
	sum_wij_email = sum(email['Weight'])
	aij_wij = pd.merge(email, hist, how='left', on=['From','To'])
	aij_wij.columns = ['From','To','Weight','Aij']
	sum_aij_hist_wij_email = sum(aij_wij[aij_wij['Aij']>0]['Weight'])
	print(sum_wij_email,sum_aij_hist_wij_email)

def get_Ci1(email_layer, hist_layer, email_graph, hist_graph):
	email_layer = email_layer[['From','To']]
	hist_layer = hist_layer[['From','To']]
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
	triangle_2_email = dict.fromkeys(list(set(email_layer['From']) | set(email_layer['To'])),0)
	del triangle_2_email['aaaa']
	triad_1_email = dict.fromkeys(list(set(email_layer['From']) | set(email_layer['To'])),0)
	del triad_1_email['aaaa']
	# triangle_2
	for key,value in email_oi_d.items():
		key_row = email_layer[email_layer['From']==key]
		for row in key_row.values.tolist():
			a = hist_layer[(hist_layer['From']==row[1]) & (hist_layer['From']!=hist_layer['To'])]
			if not a.empty:
				for hist_row in a.values.tolist():
					b = email_layer[(email_layer['From']==hist_row[1]) & \
									(email_layer['From']!=email_layer['To']) & \
									(email_layer['To']==row[0])]
					if not b.empty:
						triangle_2_email[key] += len(b)
		# triad_1
		for row in key_row.values.tolist():
			triad_1_email[key] = triad_1_email[key]+len(email_layer[(email_layer['To']==row[0]) & \
													(email_layer['From']!=row[1])])
	# hist layer
	triangle_2_hist = dict.fromkeys(list(set(hist_layer['From']) | set(hist_layer['To'])),0)
	del triangle_2_hist['aaaa']
	triad_1_hist = dict.fromkeys(list(set(hist_layer['From']) | set(hist_layer['To'])),0)
	del triad_1_hist['aaaa']
	for key,value in hist_oi_d.items():
		key_row = hist_layer[hist_layer['From']==key]
		for row in key_row.values.tolist():
			a = email_layer[(email_layer['From']==row[1]) & (email_layer['From']!=email_layer['To'])]
			if not a.empty:
				for email_row in a.values.tolist():
					b = hist_layer[(hist_layer['From']==email_row[1]) & \
									(hist_layer['From']!=hist_layer['To']) & \
									(hist_layer['To']==row[0])]
					if not b.empty:
						triangle_2_hist[key] += len(b)
		for row in key_row.values.tolist():
			triad_1_hist[key] = triad_1_hist[key]+len(hist_layer[(hist_layer['To']==row[0]) & \
													(hist_layer['From']!=row[1])])
	triangle_2 = dict(Counter(triangle_2_email)+Counter(triangle_2_hist))
	triad_1 = dict(Counter(triad_1_email)+Counter(triad_1_hist))
	Ci1 = {}
	for i in triangle_2.keys():
		Ci1[i] = round(triangle_2[i]/float(triad_1[i]),2)
	return Ci1 # not include no link nodes

def aggregated_network(email, hist):
 	agg_network = pd.concat([email,hist])
 	agg_network = agg_network.sort_values(['Weight'], ascending=True).drop_duplicates(['From','To'],keep='first')
 	agg_network = agg_network[~(agg_network['From'] == agg_network['To'])]
	return agg_network

def get_lambdai(email_layer, hist_layer, email_graph, hist_graph):
	# aggregate network
	df = aggregated_network(email_layer,hist_layer)
	agg_G = nx.DiGraph()
	agg_G = agg_G.to_directed()
	triad = list(zip(*[df[c].values.tolist() for c in df]))
	agg_G.add_weighted_edges_from(triad)
	agg_G.remove_node('aaaa')
	# total number of shortest paths on multiplex network
	agg_path   = nx.all_pairs_dijkstra_path(agg_G)
	email_path = nx.all_pairs_dijkstra_path(email_graph)
	hist_path  = nx.all_pairs_dijkstra_path(hist_graph)
	agg_dict, email_dict, hist_dict = {}, {}, {}
	email_edge_path_dict, hist_edge_path_dict, agg_edge_path_dict = {}, {}, {} # shortest paths of all nodes
	lambdai = {}
	for node,path in email_path:
		for i in path:
			email_edge_path_dict['++'.join(path[i])] = 1
	for node,path  in hist_path:
		for i in path:
			hist_edge_path_dict['++'.join(path[i])] = 1
	for node,path  in agg_path:
		for i in path:
			agg_edge_path_dict['++'.join(path[i])] = 1
	# count the shortest path of node i in agg_network
	bottom_df = pd.DataFrame(agg_edge_path_dict.items(),columns=['edge','count'])
	bottom_df['From'] = bottom_df['edge'].map(lambda x:x.split('++')[0])
	bottom_count = Counter(bottom_df['From'])
	for edge in email_edge_path_dict:
		if agg_edge_path_dict.get(edge, 'aaaa') != 'aaaa':
			agg_edge_path_dict.pop(edge)
	# count the shortest path of all nodes between two layers
	for edge in hist_edge_path_dict:
		if agg_edge_path_dict.get(edge, 'aaaa') != 'aaaa':
			agg_edge_path_dict.pop(edge)
	# count the shortest path of node i  between two layers
	top_df = pd.DataFrame(agg_edge_path_dict.items(),columns=['edge','count'])
	top_df['From'] = top_df['edge'].map(lambda x:x.split('++')[0])
	top_count = Counter(top_df['From'])
	# calculate the lambda i
	lambdai = {}
	for k in bottom_count:
		if top_count.get(k,'aaaa') != 'aaaa':
			lambdai[k] = round(top_count[k]/float(bottom_count[k]),2)
	return lambdai

def get_eigenvector(email,hist):
	email_eigen_cen = nx.eigenvector_centrality(email)
	hist_eigen_cen  = nx.eigenvector_centrality(hist)
	# multiplex network eigenvector centrality
	nodelist_email = sorted(list(email.nodes()))
	nodelist_hist  = sorted(list(hist.nodes()))
	adj_email = nx.to_numpy_matrix(email, nodelist=nodelist_email, weight='weight', nonedge=0.0)
	adj_hist  = nx.to_numpy_matrix(hist , nodelist=nodelist_hist, weight='weight', nonedge=0.0)
	b = 0.5
	Mb = b * adj_email + (1-b) * adj_hist
	M_graph = nx.DiGraph()
	M_graph = M_graph.to_directed()
	for row_index in range(Mb.shape[0]):
		for col_index in range(Mb.shape[1]):
			M_graph.add_edge(row_index, col_index, weight=Mb[row_index,col_index])
	M_eigen_cen = nx.eigenvector_centrality(M_graph)
	print(M_eigen_cen)

def multiplex_network_property(graph, layer):
	# o_ij 
	# print("overlap of eclipse edges", len(set(graph['ee'].edges()) & set(graph['eh'].edges())))
	# print("overlap of gnome edges", len(set(graph['ge'].edges()) & set(graph['gh'].edges())))

	# o_i
	# dict2csv(get_oi(graph['ee'],graph['eh']), './result/oi_eclipse.csv')
	# print("eclipse degree overlap of node i")
	# dict2csv(get_oi(graph['ge'],graph['gh']), './result/oi_gnome.csv')
	# print("gnome   degree overlap of node i")

	# o_ij_w,not all node have a overlap weight
	# get_o_ij_w(layer['ee'], layer['eh']).to_csv('./result/oij_w_eclipse.csv', index=None)
	# print("eclipse weight overlap of link ij finished")
	# get_o_ij_w(layer['ge'], layer['gh']).to_csv('./result/oij_w_gnome.csv', index=None)
	# print("gnome   weight overlap of link ij finished")

	# H_i,the node distribution of different layer
	# oi = get_oi(graph['ee'],graph['eh'])
	# dict2csv(get_cross_entropy(graph['ee'], graph['eh'], oi), './result/cross_entropy_eclipse.csv')
	# print("eclipse entropy of the multiplex degree finished")
	# dict2csv(get_cross_entropy(graph['ge'], graph['gh'], oi), './result/cross_entropy_gnome.csv')
	# print("gnome   entropy of the multiplex degree finished")

	# pw_aijHist_wij
	# print("eclipse conditional probability of finding \
	# 		a link at hist layer given the presence of \
	# 		an weight edge betweene the same nodes at email layer",get_pw(layer['ee'],layer['eh']))
	# print("gnome   conditional probability of finding \
	# 		a link at hist layer given the presence of \
	# 		an weight edge betweene the same nodes at email layer",get_pw(layer['ge'],layer['gh']))

	# Ci1
	# dict2csv(get_Ci1(layer['ee'],layer['eh'],graph['ee'],graph['eh']), './result/Ci1_eclipse.csv')
	# print("eclipse first coefficient Ci1 finished")
	# dict2csv(get_Ci1(layer['ge'],layer['gh'],graph['ge'],graph['gh']), './result/Ci1_gnome.csv')
	# print("  gnome first coefficient Ci1 finished")

	# lambda i
	# dict2csv(get_lambdai(layer['ee'],layer['eh'],graph['ee'],graph['eh']), './result/lambdai_eclipse.csv')
	# print("eclipse interdependence lambda i finished")
	# dict2csv(get_lambdai(layer['ge'],layer['gh'],graph['ge'],graph['gh']), './result/lambdai_gnome.csv')
	# print("gnome   interdependence lambda i finished")

	# eigenvector centrility
	# print("eclipse eigenvector centrility:", get_eigenvector(graph['ee'],graph['eh']))
	# print("gnome   eigenvector centrility:", get_eigenvector(graph['ge'],graph['gh']))
	pass

def main():
	# preprocessing, drop_duplicates，delete From==NA and To==NA
	# for root, dirs, files in os.walk('./raw_data/'):
	# 	preprocess(files)

	# alignment the devs of multiplex network
	# platform = ('eclipse', 'gnome')
	# layer = ('history', 'email')
	# common_node(platform, layer)

	layer_data = load_data()

	# calculate the degree、closeness、betweenness of single layer, print the same dev at topK
	# single_layer_property(layer_data)

	# build a graph by using networkx
	graph = build_graph(layer_data)

	# multiplex network properties, http://xueshu.baidu.com/s?wd=paperuri%3A%28c50f99f17d678bd37364f5c3bf863941%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1308.3182&ie=utf-8&sc_us=5600601886595482789
	multiplex_network_property(graph, layer_data)

if __name__ == '__main__':
	main()