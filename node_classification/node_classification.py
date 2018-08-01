# -*- coding: utf-8-*-
import pandas as pd
import os
import csv

def index2str(index_data):
	name = sorted(list(set(index_data.From) | set(index_data.To)))
	index = range(len(name))
	name_index = dict(zip(name, index))
	return name_index
	

# eclipse
def bug_fix_path(bug_path, name_index):
	bug_id_path = dict() # bugid + 修复路径
	for maindir, subdir, file_name_list in os.walk(bug_path):
		for filename in file_name_list:
			apath = os.path.join(maindir, filename)
			bug_id = filename.split('.')[0]
			ass_cc = []
			with open(apath) as csvfile:
				bug_history = csv.reader(csvfile)
				for row in bug_history:
					if (len(row) == 5) and ((row[2] == 'Assignee') or (row[2] == 'CC')): # eclipse gnome
						ass_cc.append(row)
			ass_cc_df = pd.DataFrame(ass_cc, columns=['When', 'Who', 'What', 'Added', 'Removed']) # eclipse
			history_name_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
			history_name = set(map(lambda x: x.split('@')[0], history_name_list)) # eclipse
			if set(name_index) & history_name:
				tmp_path_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
				tmp_path = list(map(lambda x:x.split('@')[0], tmp_path_list)) # eclipse
				bug_id_path[bug_id] = tmp_path
	return bug_id_path


# gnome
# def bug_fix_path(bug_path, name_index):
# 	bug_id_path = dict() # bugid + 修复路径
# 	for maindir, subdir, file_name_list in os.walk(bug_path):
# 		for filename in file_name_list:
# 			apath = os.path.join(maindir, filename)
# 			bug_id = filename.split('.')[0].split('_')[1]
# 			ass_cc = []
# 			with open(apath) as csvfile:
# 				bug_history = csv.reader(csvfile)
# 				for row in bug_history:
# 					if (len(row) == 5) and ((row[2] == 'Assignee') or (row[2] == 'CC')): # eclipse gnome
# 						ass_cc.append(row)
# 			ass_cc_df = pd.DataFrame(ass_cc, columns=['Who', 'When', 'What', 'Removed', 'Added']) # gnome
# 			history_name_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
# 			history_name = set(map(lambda x:x.replace('@', ' ').replace('.', ' '), history_name_list)) # gnome
# 			if set(name_index) & history_name:
# 				tmp_path_list = list(ass_cc_df['Who']) + list(ass_cc_df['Added'])
# 				tmp_path = list(map(lambda x:x.replace('@', ' ').replace('.', ' '), tmp_path_list)) # gnome
# 				bug_id_path[bug_id] = tmp_path
# 	return bug_id_path


def count_bug(bugid_path):
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
	count_dev_df_T = count_dev_df.stack(level=0)
	print(count_dev_df_T)


def dict2csv(_dict, file):
	print(_dict)
	with open(file, 'wb') as f:
		w = csv.writer(f)
			

def main():
	# filename = ('eclipsehistory.csv', 'gnomehistory.csv')
	# index_data = pd.read_csv('./index_data/' + filename[0]) # eclise
	# # index_data = pd.read_csv('./index_data/' + filename[1]) # gnome
	# name_index = index2str(index_data)

	# # bug_path = '/home/mia/Desktop/eclipse_bug_history/' # test
	# # bug_path = '/media/mia/文档/wubo/eclipse_data/eclipse_history/原始/bughistory_raw/' # eclipse
	# # bug_path = '/media/mia/文档/wubo/Gnome_data/Gnome_History/Gnome_History_select/' # gnome
	# bug_id_path = bug_fix_path(bug_path, name_index)

	# for k, v in bug_id_path.items():
	# 	bug_id_path[k] = '++'.join(v)
	# bug_id_path_df = pd.DataFrame.from_dict(bug_id_path, orient='index')
	# bug_id_path_df.reset_index(inplace=True)
	# bug_id_path_df.columns = ['bugid', 'path']
	# bug_id_path_df.to_csv('./eclipse_bugid_path.csv', index=False) # eclipse
	# bug_id_path_df.to_csv('./gnome_bugid_path.csv', index=False) # gnome

	bug_id_path = pd.read_csv('eclipse_bugid_path.csv') # eclipse
	# bug_id_path = pd.read_csv('gnome_bugid_path.csv') # gnome
	count_bug(bug_id_path)
	


if __name__ == '__main__':
	main()