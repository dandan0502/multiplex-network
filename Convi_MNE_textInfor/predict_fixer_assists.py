import pandas as pd
import numpy as np

filename = 'gnome' # gnome

text_path = "./textInfo/"
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
	left_df.drop(['fixers'], axis=1, inplace=True)
	left_list = np.array(left_df)
	right_df.drop(['idx'], axis=1, inplace=True)
	right_list = np.array(right_df)
	# 2.calculate cos similarity
	left_list, right_list = normalize(left_list), normalize(right_list)
	results = left_list.dot(right_list.T)
	# 3.get results as forming left index,right index,result
	results_df = pd.DataFrame(results.flatten())
	all_idx = []
	for b in left_idx:
		for d in right_idx:
			all_idx.append([b, d])
	all_idx_df = pd.DataFrame(all_idx)
	cos_similarity_df = pd.concat([all_idx_df, results_df], axis=1)
	cos_similarity_df.columns = ['bugid', 'tosser', 'cos']
	# 4.get top n cos value of each bugid
	topn_df = cos_similarity_df.groupby('bugid').apply(lambda x:x.sort_values('cos')[-n:]).reset_index(drop=True)
	return topn_df


def structure():
	# 1.concat all vectors of fixers, bugs and all nodes
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_100.csv")
	other_structure = pd.read_csv(structure_path + filename + "_emb_result_200.csv", header=None)
	fixer_structure = pd.read_csv(structure_path + filename + "_emb_result_100.csv", header=None)
	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	other_structure.columns = ['idx'] + list(range(other_structure.shape[1] - 1))
	fixer_structure.columns = ['fixers'] + list(range(fixer_structure.shape[1] - 1))
	# structures + bug_des_com = 200
	bugid_fixer_structure = pd.merge(bugid_fixer, fixer_structure, how='inner', on='fixers')
	bugid_fixer_structure_des_com = pd.merge(bugid_fixer_structure, bug_des_com, how='inner', on='bugid')
	bugid_fixer_structure_des_com['fixers'] = bugid_fixer_structure_des_com['fixers'].astype('str')
	bugid_fixer_structure_des_com['bugid'] = bugid_fixer_structure_des_com['bugid'].astype('str')
	# 2.count cos similarity
	bugid_fixer_structure_des_com['fixers'] = bugid_fixer_structure_des_com['fixers'] + '++' + bugid_fixer_structure_des_com['bugid']
	fixer_bug_df = bugid_fixer_structure_des_com.drop(['bugid', 'tossers'], axis=1)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, other_structure, 100)
	# 3.count structure acc
	acc = count_tosser_acc(top10_tosser_df)


def text_info():
	# 1.concat all vectors of dev
	other_structure = pd.read_csv(structure_path + filename + "_emb_result_200.csv", header=None)
	other_structure.columns = ['idx'] + list(range(other_structure.shape[1] - 1))

	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_68.csv")
	dev_com = pd.read_csv(text_path + filename + "_fixer_component51.csv")
	dev_pro = pd.read_csv(text_path + filename + "_fixer_product51.csv")
	dev_abs = pd.read_csv(text_path + filename + "_fixer_abstract.csv")
	dev_com = dev_com.fillna(0)
	dev_com = dev_com.drop('fixers', axis=1)
	dev_pro = dev_pro.fillna(0)
	dev_pro = dev_pro.drop('fixers', axis=1)
	dev_vec = pd.concat([dev_abs, dev_com, dev_pro], axis=1)
	dev_vec.to_csv(text_path + filename + "_fixer_textInfo.csv", index=False)

	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	bugid_fixer_text = pd.merge(bugid_fixer, dev_vec, how='inner', on='fixers')
	bugid_fixer_text_des_com = pd.merge(bugid_fixer_text, bug_des_com, how='inner', on='bugid')
	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'].astype('str')
	bugid_fixer_text_des_com['bugid'] = bugid_fixer_text_des_com['bugid'].astype('str')

	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'] + '++' + bugid_fixer_text_des_com['bugid']
	fixer_bug_df = bugid_fixer_text_des_com.drop(['bugid', 'tossers'], axis=1)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, other_structure, 20)
	acc = count_tosser_acc(top10_tosser_df)


def all_info():
	other_structure = pd.read_csv(structure_path + filename + "_emb_result_300.csv", header=None)
	other_structure.columns = ['idx'] + list(range(other_structure.shape[1] - 1))
	# 1.concat text vectors and structure vectors
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_68.csv")
	dev_text = pd.read_csv(text_path + filename + "_fixer_textInfo.csv")
	dev_strc = pd.read_csv(structure_path + filename + "_emb_result_100.csv", header=None)
	dev_strc.columns = ['fixers'] + list(range(dev_strc.shape[1] - 1))
	dev_vec = pd.merge(dev_text, dev_strc, how='inner', on='fixers')

	bugid_fixer = bug_tosser_fixer.drop(['last_t', 'other_t'], axis=1)
	bugid_fixer_text = pd.merge(bugid_fixer, dev_vec, how='inner', on='fixers')
	bugid_fixer_text_des_com = pd.merge(bugid_fixer_text, bug_des_com, how='inner', on='bugid')
	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'].astype('str')
	bugid_fixer_text_des_com['bugid'] = bugid_fixer_text_des_com['bugid'].astype('str')

	bugid_fixer_text_des_com['fixers'] = bugid_fixer_text_des_com['fixers'] + '++' + bugid_fixer_text_des_com['bugid']
	fixer_bug_df = bugid_fixer_text_des_com.drop(['bugid', 'tossers'], axis=1)
	top10_tosser_df = topn_cos_similarity(fixer_bug_df, other_structure, 100)
	acc = count_tosser_acc(top10_tosser_df)


def get_top1_tosser_acc(l):
	if l['tosser'] in l['tossers'].split('++'):
		l['top1'] = 1
	else:
		l['top1'] = 0
	return l


def count_tosser_acc(topn_df):
	bug_tosser_fixer['fixers'] = bug_tosser_fixer['fixers'].astype('str')
	bug_tosser_fixer['bugid'] = bug_tosser_fixer['bugid'].astype('str')
	bug_tosser_fixer['bugid'] = bug_tosser_fixer['fixers'] + '++' + bug_tosser_fixer['bugid']
	# 1.top1 acc
	# top1_df = topn_df.drop_duplicates(['bugid'], keep='last')
	# fixerbugid_tosser = pd.merge(top1_df, bug_tosser_fixer, how='inner', on='bugid')
	# fixerbugid_tosser_top1 = fixerbugid_tosser.apply(get_top1_tosser_acc, axis=1)
	# print(fixerbugid_tosser_top1['top1'].sum() / fixerbugid_tosser_top1.shape[1])
	# 2. label top 10 
	# print(topn_df, bug_tosser_fixer)
	fixerbugid_tosser = pd.merge(topn_df, bug_tosser_fixer, how='inner', on='bugid')
	fixerbugid_tosser_topn = fixerbugid_tosser.apply(get_top1_tosser_acc, axis=1)
	# 3.print acc
	print(fixerbugid_tosser_topn['top1'].sum() , fixerbugid_tosser_topn.shape[1])


def main():
	# 1.get bug-fixer-tosser's dict
	global bug_tosser_fixer_dict
	global bug_tosser_fixer
	bug_tosser_fixer = pd.read_csv(fix_path + filename + "_bugid_tossers_fixers.csv")
	bug_tosser_fixer = bug_tosser_fixer.apply(get_other_tosser, axis=1)
	bug_tosser_fixer_dict = bug_tosser_fixer.set_index('bugid').to_dict('index')
	# 2.structure - 100
	structure()
	# 3.text - 132
	# text_info()
	# all
	# all_info()


if __name__ == '__main__':
	main()