import pandas as pd
import numpy as np

filename = 'gnome' # gnome

# ------------------------------------
# text_path = "./textInfo/"
text_path = "./textInfo_delCom/"
# ------------------------------------
fix_path = "./assists_dev_MNE/"
structure_path = "./Convi_MNE/emb_result/"


def get_other_tosser(l):
	tossers = l['tossers'].split('++')
	if len(tossers) > 1:
		l['last_t'] = tossers[-1]
		l['other_t'] = '++'.join(tossers[:-1])
	else:
		l['last_t'] = tossers[-1]
		l['other_t'] = tossers[-1]
	return l


def normalize(A):
	lengths = (A**2).sum(axis=1, keepdims=True)**.5
	return A / lengths


def topn_cos_similarity(left_df, right_df, n):
	# 1.get two np.array to calculate
	left_idx = list(left_df['fixers'])
	right_idx = list(right_df['idx'])
	left_df = left_df.drop(['fixers'], axis=1)
	left_list = np.array(left_df)
	right_df = right_df.drop(['idx'], axis=1)
	right_list = np.array(right_df)
	# 2.calculate cos similarity
	left_list, right_list = normalize(left_list), normalize(right_list)
	results = []

	rows_in_slice = 100

	slice_start = 0
	slice_end = slice_start + rows_in_slice

	while slice_end <= left_list.shape[0]:
		results.append(left_list[slice_start:slice_end].dot(right_list.T))
		slice_start += rows_in_slice
		slice_end = slice_start + rows_in_slice
	results = np.concatenate(results)
	# results = left_list.dot(right_list.T)
	# 3.get results as forming left index,right index,result
	del left_df, right_df, left_list, right_list
	results_df = pd.DataFrame(results.flatten())
	print('finish dot')
	all_idx = []
	for b in left_idx:
		for d in right_idx:
			all_idx.append([b, d])
	all_idx_df = pd.DataFrame(all_idx)
	cos_similarity_df = pd.concat([all_idx_df, results_df], axis=1)
	cos_similarity_df.columns = ['bugid', 'tosser', 'cos']
	print('finish idx')
	# 4.get top n cos value of each bugid
	group_cos = cos_similarity_df.groupby('bugid')
	print('finish groupby')
	topn_df = pd.DataFrame()
	group_list = []
	for name, group in group_cos:
		group_list.append(group.sort_values('cos')[-n:])
	topn_df = pd.concat(group_list, axis=0)
	return topn_df


def structure():
	# 1.concat all vectors of fixers, bugs and all nodes
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_200.csv")
	fixer_structure = pd.read_csv(structure_path + filename + "_emb_result_50.csv", header=None)
	fixer_structure.columns = ['fixers'] + list(range(fixer_structure.shape[1] - 1))

	dev_com = pd.read_csv(text_path + filename + "_fixer_component51.csv")
	dev_pro = pd.read_csv(text_path + filename + "_fixer_product51.csv")
	dev_abs = pd.read_csv(text_path + filename + "_fixer_abstract_98.csv")
	dev_com = dev_com.fillna(0)
	dev_com = dev_com.drop('fixers', axis=1)
	dev_pro = dev_pro.fillna(0)
	dev_pro = dev_pro.drop('fixers', axis=1)
	dev_vec = pd.concat([dev_abs, dev_com, dev_pro], axis=1)
	# dev_vec.to_csv(text_path + filename + "_fixer_textInfo.csv", index=False)

	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	dev_structure_vec = pd.merge(dev_vec, fixer_structure, how='inner', on='fixers')
	dev_structure_vec.columns = ['idx'] + list(range(dev_structure_vec.shape[1] - 1))
	# structures + bug_des_com = 200
	bugid_fixer_structure = pd.merge(bugid_fixer, fixer_structure, how='inner', on='fixers')
	bugid_fixer_structure_des_com = pd.merge(bugid_fixer_structure, bug_des_com, how='inner', on='bugid')
	bugid_fixer_structure_des_com['fixers'] = bugid_fixer_structure_des_com['fixers'].astype('str')
	bugid_fixer_structure_des_com['bugid'] = bugid_fixer_structure_des_com['bugid'].astype('str')
	# 2.count cos similarity
	bugid_fixer_structure_des_com['fixers'] = bugid_fixer_structure_des_com['fixers'] + '++' + bugid_fixer_structure_des_com['bugid']
	fixer_bug_df = bugid_fixer_structure_des_com.drop(['bugid', 'tossers'], axis=1)
	# top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_structure_vec, 3)
	# acc = count_tosser_acc(top10_tosser_df)

	top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_structure_vec, 5)
	# 3.count structure acc
	acc = count_tosser_acc(top10_tosser_df)
	# print('finish acc')


def struc_sem():
	# left
	fixer_structure = pd.read_csv(structure_path + filename + "_emb_result_200.csv", header=None)
	fixer_structure.columns = ['fixers'] + list(range(fixer_structure.shape[1] - 1))
	fixer_structure['fixers'] = fixer_structure['fixers'].astype('str')
	bug_all_text = pd.read_csv(text_path + filename + "_bugs_all_text_topics_200.csv")
	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	bugid_fixer['fixers'] = bugid_fixer['fixers'].astype('str')
	bugid_fixer_structure = pd.merge(bugid_fixer, fixer_structure, how='inner', on='fixers')
	bugid_fixer_structure_bug_all_text = pd.merge(bugid_fixer_structure, bug_all_text, how='inner', on='bugid')

	# right
	dev_vec = pd.read_csv(text_path + filename + "_tossers_bug_all_text_topics_200.csv")
	dev_vec.columns = ['fixers'] + list(range(dev_vec.shape[1] - 1)) # convert tossers to fixers, as to be the same with above function
	dev_vec['fixers'] = dev_vec['fixers'].astype('str')
	dev_structure_vec = pd.merge(fixer_structure, dev_vec, how='inner', on='fixers')
	dev_structure_vec.columns = ['idx'] + list(range(dev_structure_vec.shape[1] - 1))

	# print(bugid_fixer_structure_bug_all_text.columns, dev_structure_vec.columns)

	# evaluate
	bugid_fixer_structure_bug_all_text['fixers'] = bugid_fixer_structure_bug_all_text['fixers'].astype('str')
	bugid_fixer_structure_bug_all_text['bugid'] = bugid_fixer_structure_bug_all_text['bugid'].astype('str')
	bugid_fixer_structure_bug_all_text['fixers'] = bugid_fixer_structure_bug_all_text['fixers'] + '++' + bugid_fixer_structure_bug_all_text['bugid']
	fixer_bug_df = bugid_fixer_structure_bug_all_text.drop(['bugid', 'tossers'], axis=1)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_structure_vec, 10)
	acc = count_tosser_acc(top10_tosser_df)


def text_info():
	# 1.concat all vectors of dev
	# right
	dev_com = pd.read_csv(text_path + filename + "_fixer_component151.csv")
	dev_pro = pd.read_csv(text_path + filename + "_fixer_product151.csv")
	dev_abs = pd.read_csv(text_path + filename + "_fixer_abstract_198.csv")
	dev_com = dev_com.fillna(0)
	dev_com = dev_com.drop('fixers', axis=1)
	dev_pro = dev_pro.fillna(0)
	dev_pro = dev_pro.drop('fixers', axis=1)
	fixer_vec = pd.concat([dev_abs, dev_com, dev_pro], axis=1)
	fixer_vec.columns = ['idx'] + list(range(fixer_vec.shape[1] - 1))
	fixer_vec['idx'] = fixer_vec['idx'].astype('str')
	# fixer_vec = pd.read_csv(text_path + filename + "_fixer_textInfo.csv")
	# fixer_vec.columns = ['idx'] + list(range(fixer_vec.shape[1] - 1))

	# left
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_100.csv")
	bug_des_com['bugid'] = bug_des_com['bugid'].astype('str')
	# dev_com = pd.read_csv(text_path + filename + "_fixer_component151.csv")
	# dev_pro = pd.read_csv(text_path + filename + "_fixer_product151.csv")
	dev_abs = pd.read_csv(text_path + filename + "_fixer_abstract_98.csv")
	dev_vec = pd.concat([dev_abs, dev_com, dev_pro], axis=1)
	dev_vec.columns = ['fixers'] + list(range(dev_vec.shape[1] - 1))
	dev_vec['fixers'] = dev_vec['fixers'].astype('str')
	dev_vec.to_csv(text_path + filename + "_fixer_textInfo_400_119.csv", index=False)

	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	bugid_fixer['fixers'] = bugid_fixer['fixers'].astype('str')
	bugid_fixer_text = pd.merge(bugid_fixer, dev_vec, how='inner', on='fixers')
	bugid_fixer_text['bugid'] = bugid_fixer_text['bugid'].astype('str')
	bugid_fixer_text_des_com = pd.merge(bugid_fixer_text, bug_des_com, how='inner', on='bugid')
	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'].astype('str')
	bugid_fixer_text_des_com['bugid'] = bugid_fixer_text_des_com['bugid'].astype('str')

	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'] + '++' + bugid_fixer_text_des_com['bugid']
	fixer_bug_df = bugid_fixer_text_des_com.drop(['bugid', 'tossers'], axis=1)
	# top10_tosser_df = topn_cos_similarity(fixer_bug_df, fixer_vec, 3)
	# acc = count_tosser_acc(top10_tosser_df)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, fixer_vec, 5)
	acc = count_tosser_acc(top10_tosser_df)


def text_sem():
	# left
	fixer_text = pd.read_csv(text_path + filename + "_tossers_bug_all_text_topics_200.csv")
	fixer_text.columns = ['fixers'] + list(range(fixer_text.shape[1] - 1))
	fixer_text['fixers'] = fixer_text['fixers'].astype('str')
	bug_all_text = pd.read_csv(text_path + filename + "_bugs_all_text_topics_200.csv")
	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	bugid_fixer['fixers'] = bugid_fixer['fixers'].astype('str')
	bugid_fixer_text = pd.merge(bugid_fixer, fixer_text, how='inner', on='fixers')
	bugid_fixer_text_bug_all_text = pd.merge(bugid_fixer_text, bug_all_text, how='inner', on='bugid')

	# right
	dev_vec = pd.read_csv(text_path + filename + "_tossers_bug_all_text_topics_200.csv")
	other_dev_vec = dev_vec.drop(['tossers'], axis=1)
	dev_vec = pd.concat([dev_vec, other_dev_vec], axis=1)
	dev_vec.columns = ['idx'] + list(range(dev_vec.shape[1] - 1)) # convert tossers to idx, as to be the same with above function
	
	# print(bugid_fixer_text_bug_all_text.columns, dev_vec.columns)
	bugid_fixer_text_bug_all_text['fixers'] = bugid_fixer_text_bug_all_text['fixers'].astype('str')
	bugid_fixer_text_bug_all_text['bugid'] = bugid_fixer_text_bug_all_text['bugid'].astype('str')
	bugid_fixer_text_bug_all_text['fixers'] = bugid_fixer_text_bug_all_text['fixers'] + '++' + bugid_fixer_text_bug_all_text['bugid']
	fixer_bug_df = bugid_fixer_text_bug_all_text.drop(['bugid', 'tossers'], axis=1)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_vec, 10)
	acc = count_tosser_acc(top10_tosser_df)
	# top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_vec, 5)
	# acc = count_tosser_acc(top10_tosser_df)


def text_vecItem():
	# left
	fixer_text = pd.read_csv(text_path + filename + '_fixer_textInfo_100.csv')
	bug_text = pd.read_csv(text_path + filename + 'bugs_all_text_topics_100.csv')
	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	bugid_fixer['fixers'] = bugid_fixer['fixers'].astype('str')
	bugid_fixer_text = pd.merge(bugid_fixer, fixer_text, how='inner', on='fixers')
	bugid_fixer_text_bug_all_text = pd.merge(bugid_fixer_text, bug_text, how='inner', on='bugid')

	# right



def all_info():
	# 1.concat text vectors and structure vectors
	# left
	fixer_vec = pd.read_csv(text_path + filename + "_fixer_textInfo_400_119.csv")
	fixer_vec.columns = ['fixers'] + list(range(fixer_vec.shape[1] - 1))
	fixer_structure = pd.read_csv(structure_path + filename + "_emb_result_50.csv", header=None)
	fixer_structure.columns = ['fixers'] + list(range(fixer_structure.shape[1] - 1))
	fixer_vec = pd.merge(fixer_vec, fixer_structure, on='fixers', how='inner')
	fixer_vec['fixers'] = fixer_vec['fixers'].astype('str')
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_100.csv")
	bug_des_com['bugid'] = bug_des_com['bugid'].astype('str')

	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	bugid_fixer['fixers'] = bugid_fixer['fixers'].astype('str')
	# print(bugid_fixer['fixers'], fixer_vec['fixers'])
	bugid_fixer_text = pd.merge(bugid_fixer, fixer_vec, how='inner', on='fixers')
	bugid_fixer_text['bugid'] = bugid_fixer_text['bugid'].astype('str')
	bugid_fixer_text_des_com = pd.merge(bugid_fixer_text, bug_des_com, how='inner', on='bugid')
	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'].astype('str')
	bugid_fixer_text_des_com['bugid'] = bugid_fixer_text_des_com['bugid'].astype('str')

	# right
	dev_com = pd.read_csv(text_path + filename + "_fixer_component151.csv")
	dev_pro = pd.read_csv(text_path + filename + "_fixer_product151.csv")
	dev_abs = pd.read_csv(text_path + filename + "_fixer_abstract_248.csv")
	dev_com = dev_com.fillna(0)
	dev_com = dev_com.drop('fixers', axis=1)
	dev_pro = dev_pro.fillna(0)
	dev_pro = dev_pro.drop('fixers', axis=1)
	dev_vec = pd.concat([dev_abs, dev_com, dev_pro], axis=1)
	dev_vec.columns = ['idx'] + list(range(dev_vec.shape[1] - 1))

	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'] + '++' + bugid_fixer_text_des_com['bugid']
	fixer_bug_df = bugid_fixer_text_des_com.drop(['bugid', 'tossers'], axis=1)
	# print(fixer_bug_df, dev_vec)
	# top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_vec, 3)
	# acc = count_tosser_acc(top10_tosser_df)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_vec, 5)
	acc = count_tosser_acc(top10_tosser_df)


def all_sem():
	# left
	# fixer-structure
	fixer_structure = pd.read_csv(structure_path + filename + "_emb_result_200.csv", header=None)
	fixer_structure.columns = ['fixers'] + list(range(fixer_structure.shape[1] - 1))
	fixer_structure['fixers'] = fixer_structure['fixers'].astype('str')
	# fixer-text
	fixer_text = pd.read_csv(text_path + filename + "_tossers_bug_all_text_topics_100.csv")
	fixer_text.columns = ['fixers'] + list(range(fixer_text.shape[1] - 1))
	# bug
	bug_all_text = pd.read_csv(text_path + filename + "_bugs_all_text_topics_100.csv")
	# bug-tosser-fixer
	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)

	bugid_fixer_text = pd.merge(bugid_fixer, fixer_text, how='inner', on='fixers')
	bugid_fixer_text['fixers'] = bugid_fixer_text['fixers'].astype('str')
	structure_bugid_fixer_text = pd.merge(fixer_structure, bugid_fixer_text, how='inner', on='fixers')
	structure_bugid_fixer_text_bug_all_text = pd.merge(structure_bugid_fixer_text, bug_all_text, how='inner', on='bugid')

	# right
	tosser_vec = pd.read_csv(text_path + filename + "_tossers_bug_all_text_topics_100.csv")
	other_tosser_vec = tosser_vec.drop(['tossers'], axis=1)
	tosser_vec = pd.concat([tosser_vec, other_tosser_vec], axis=1)
	tosser_vec.columns = ['fixers'] + list(range(tosser_vec.shape[1] - 1)) # convert tossers to fixers, as to be the same with above function
	tosser_vec['fixers'] = tosser_vec['fixers'].astype('str')
	dev_vec = pd.merge(fixer_structure, tosser_vec, on='fixers', how='inner')
	dev_vec.columns = ['idx'] + list(range(dev_vec.shape[1] - 1))

	# print(structure_bugid_fixer_text_bug_all_text.columns, dev_vec.columns)
	structure_bugid_fixer_text_bug_all_text['fixers'] = structure_bugid_fixer_text_bug_all_text['fixers'].astype('str')
	structure_bugid_fixer_text_bug_all_text['bugid'] = structure_bugid_fixer_text_bug_all_text['bugid'].astype('str')
	structure_bugid_fixer_text_bug_all_text['fixers'] = structure_bugid_fixer_text_bug_all_text['fixers'] + '++' + structure_bugid_fixer_text_bug_all_text['bugid']
	fixer_bug_df = structure_bugid_fixer_text_bug_all_text.drop(['bugid', 'tossers'], axis=1)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_vec, 10)
	acc = count_tosser_acc(top10_tosser_df)
	# top10_tosser_df = topn_cos_similarity(fixer_bug_df, dev_vec, 5)
	# acc = count_tosser_acc(top10_tosser_df)


def get_top1_tosser_acc(l):
	if str(l['tosser']) in l['tossers'].split('++'):
		l['top1'] = 1
	else:
		l['top1'] = 0
	return l


def count_tosser_acc(topn_df):
	bug_tosser_fixer['fixers'] = bug_tosser_fixer['fixers'].astype('str')
	bug_tosser_fixer['bugid'] = bug_tosser_fixer['bugid'].astype('str')
	bug_tosser_fixer['bugid'] = bug_tosser_fixer['fixers'] + '++' + bug_tosser_fixer['bugid']
	# 1.top1 acc
	top1_df = topn_df.drop_duplicates(['bugid'], keep='last')
	fixerbugid_tosser = pd.merge(top1_df, bug_tosser_fixer, how='inner', on='bugid')
	fixerbugid_tosser_top1 = fixerbugid_tosser.apply(get_top1_tosser_acc, axis=1)
	# print(fixerbugid_tosser_top1)
	print(fixerbugid_tosser_top1['top1'].sum() , fixerbugid_tosser_top1.shape[0])
	# 2. label top 10 
	fixerbugid_tosser = pd.merge(topn_df, bug_tosser_fixer, how='inner', on='bugid')
	fixerbugid_tosser_topn = fixerbugid_tosser.apply(get_top1_tosser_acc, axis=1)
	# 3.print acc
	# print(fixerbugid_tosser_topn['top1'])
	print(fixerbugid_tosser_topn['top1'].sum() , fixerbugid_tosser_top1.shape[0])


def main():
	# 1.get bug-fixer-tosser's dict
	global bug_tosser_fixer_dict
	global bug_tosser_fixer
	bug_tosser_fixer = pd.read_csv(fix_path + filename + "_bugid_tossers_fixers.csv")
	bug_tosser_fixer = bug_tosser_fixer.apply(get_other_tosser, axis=1)
	bug_tosser_fixer_dict = bug_tosser_fixer.set_index('bugid').to_dict('index')
	# 2.structure 
	# structure()
	# 3.text 
	# text_info()
	# 4.all
	# all_info()
	# 5.same semantic space
	# struc_sem()
	# text_sem()
	all_sem()
	# 6.different semantic space but same vectors item
	# text_vecItem()
	# all_vecItem()


if __name__ == '__main__':
	main()