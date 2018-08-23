# -*- coding: utf-8-*-
from __future__ import division

import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import math


def index2str(index_data):
	name = sorted(list(set(index_data.From) | set(index_data.To)))
	index = range(len(name))
	name_index = dict(zip(name, index))
	return name_index
	

# eclipse
# def bug_fix_path(bug_path, name_index):
# 	bug_id_path = dict() # bugid + 修复路径
# 	for maindir, subdir, file_name_list in os.walk(bug_path):
# 		for filename in file_name_list:
# 			apath = os.path.join(maindir, filename)
# 			bug_id = filename.split('.')[0]
# 			ass_cc = []
# 			with open(apath) as csvfile:
# 				bug_history = csv.reader(csvfile)
# 				for row in bug_history:
# 					if (len(row) == 5) and ((row[2] == 'Assignee') or (row[2] == 'CC')): # eclipse gnome
# 						ass_cc.append(row)
# 			ass_cc_df = pd.DataFrame(ass_cc, columns=['When', 'Who', 'What', 'Added', 'Removed']) # eclipse
# 			history_name_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
# 			history_name = set(map(lambda x: x.split('@')[0], history_name_list)) # eclipse
# 			if set(name_index) & history_name:
# 				tmp_path_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
# 				tmp_path = list(map(lambda x:x.split('@')[0], tmp_path_list)) # eclipse
# 				bug_id_path[bug_id] = tmp_path
# 	return bug_id_path


# gnome
def bug_fix_path(bug_path, name_index):
	bug_id_path = dict() # bugid + 修复路径
	for maindir, subdir, file_name_list in os.walk(bug_path):
		for filename in file_name_list:
			apath = os.path.join(maindir, filename)
			bug_id = filename.split('.')[0].split('_')[1]
			ass_cc = []
			with open(apath) as csvfile:
				bug_history = csv.reader(csvfile)
				for row in bug_history:
					if (len(row) == 5) and ((row[2] == 'Assignee') or (row[2] == 'CC')): # eclipse gnome
						ass_cc.append(row)
			ass_cc_df = pd.DataFrame(ass_cc, columns=['Who', 'When', 'What', 'Removed', 'Added']) # gnome
			history_name_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
			history_name = set(map(lambda x:x.replace('@', ' ').replace('.', ' '), history_name_list)) # gnome
			if set(name_index) & history_name:
				tmp_path_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
				tmp_path = list(map(lambda x:x.replace('@', ' ').replace('.', ' '), tmp_path_list)) # gnome
				bug_id_path[bug_id] = tmp_path
	return bug_id_path


def is_tosser(row):
	if row[1] and (not row[2]) and (not row[3]):
		row['tosser'] = 1
	else:
		row['tosser'] = 0
	return row


def count_fix_percent(row):
	row['fix_percent'] = round(row[2] / float(row[1] + row[2] + row[3]), 3)
	return row


def count_bug(bugid_path, name_index):
	count_dev = {'begin' : {}, 'end' : {}, 'middle' : {}} # 存放所有开发者的行为统计记录
	for path in bugid_path['path']:
		path_list = path.split('++')
		tmp_begin, tmp_end = path_list[0], path_list[-1]
		# count_history = {} 
		if count_dev['begin'].has_key(tmp_begin):
			count_dev['begin'][tmp_begin] += 1
		else:
			count_dev['begin'][tmp_begin] = 1
		if count_dev['end'].has_key(tmp_end):
			count_dev['end'][tmp_end] += 1
		else:
			count_dev['end'][tmp_end] = 1
		for middle_dev in path_list[1:-1]:
			if count_dev['middle'].has_key(middle_dev):
				count_dev['middle'][middle_dev] += 1
			else:
				count_dev['middle'][middle_dev] = 1
	count_dev_df = pd.DataFrame.from_dict(count_dev, orient='index')
	count_dev_df_T = count_dev_df.T
	# count_dev_df_T = count_dev_df_T.drop('') # eclipse
	count_dev_df_T.reset_index(inplace=True)
	count_dev_df_T = count_dev_df_T.fillna(0)
	count_dev_df_T = count_dev_df_T.apply(is_tosser, axis=1)
	count_dev_df_T = count_dev_df_T.apply(count_fix_percent, axis=1)
	count_dev_df_T.rename(columns={'index':'name'}, inplace = True)
	
	name_index_df = pd.DataFrame.from_dict(name_index, orient='index')
	name_index_df.reset_index(inplace=True)
	name_index_df.columns = ['name', 'index']
	name_index_percent = pd.merge(name_index_df, count_dev_df_T, on='name', how='left')
	return name_index_percent


def node_label(row):
	row['label'] = 1 if row['fix_percent'] >= row['threshold'] else 0
	return row
			

def main():
	# 把公共开发者的名字string和index对照
	# filename = ('eclipsehistory.csv', 'gnomehistory.csv')
	# # index_data = pd.read_csv('./index_data/' + filename[0]) # eclise
	# index_data = pd.read_csv('./index_data/' + filename[1]) # gnome
	# name_index = index2str(index_data)

	# 从原始数据中选出bug修复路径
	# # bug_path = '/home/mia/Desktop/eclipse_bug_history/' # test
	# # bug_path = '/media/mia/文档/wubo/eclipse_data/eclipse_history/原始/bughistory_raw/' # eclipse
	# # bug_path = '/media/mia/文档/wubo/Gnome_data/Gnome_History/Gnome_History_select/' # gnome
	# bug_id_path = bug_fix_path(bug_path, name_index)

	# 把bug修复路径存入文件中
	# for k, v in bug_id_path.items():
	# 	bug_id_path[k] = '++'.join(v)
	# bug_id_path_df = pd.DataFrame.from_dict(bug_id_path, orient='index')
	# bug_id_path_df.reset_index(inplace=True)
	# bug_id_path_df.columns = ['bugid', 'path']
	# bug_id_path_df.to_csv('./eclipse_bugid_path.csv', index=False) # eclipse
	# bug_id_path_df.to_csv('./gnome_bugid_path.csv', index=False) # gnome

	# 计算公共开发者的bug修复率并写入文件
	# bug_id_path = pd.read_csv('eclipse_bugid_path.csv') # eclipse
	# bug_id_path = pd.read_csv('gnome_bugid_path.csv') # gnome
	# name_index_percent = count_bug(bug_id_path, name_index)
	# name_index_percent.to_csv('./gnome_name_index_percent.csv', index=False)

	# 查看bug修复率的分布
	# index_percent = pd.read_csv('./eclipse_name_index_percent.csv') # eclipse
	# index_percent = pd.read_csv('./gnome_name_index_percent.csv') # gnome
	# index_percent.loc[:, 'fix_percent_group'] = index_percent.loc[:, 'fix_percent'].map(lambda x:math.ceil(x*10)/10)
	# new_df = index_percent.groupby(['fix_percent_group'])['fix_percent_group'].agg({'pindu':lambda x:len(x)}).reset_index()
	# print(new_df)
	# x, y = new_df.loc[:, 'fix_percent_group'], new_df.loc[:, 'pindu']
	# plt.scatter(x, y)
	# plt.show()

	# 对数据打标签
	# index_percent = pd.read_csv('./eclipse_name_index_percent.csv') # eclipse
	index_percent = pd.read_csv('./gnome_name_index_percent.csv') # gnome
	for threshold in [0.5, 0.8, 0.99]:
		index_percent['threshold'] = threshold
		index_percent = index_percent.fillna(0)
		index_percent1 = index_percent.apply(node_label, axis=1)
		index_percent1 = index_percent1.loc[:, ['index', 'label']]
		# index_percent1.to_csv('./threshold_data/eclipse_{}.csv'.format(threshold), index=False) # eclipse
		index_percent1.to_csv('./threshold_data/gnome_{}.csv'.format(threshold), index=False) # gnome


if __name__ == '__main__':
	main()