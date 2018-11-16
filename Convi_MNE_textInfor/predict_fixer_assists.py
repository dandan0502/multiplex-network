import pandas as pd
import numpy as np

filename = 'eclipse' # gnome

text_path = "./textInfo/"
fix_path = "./assists_dev_MNE/"
structure_path = "./Convi_MNE/emb_result/"

def normalize(A):
	lengths = (A**2).sum(axis=1, keepdims=True)**.5
	return A / lengths


def top3_cos_similarity(bug_des_com_list, dev_structure_list, bugid, dev_id):
	# cos_similarity = []
	# for index_bug, b in enumerate(bugid):
	# 	bug_vector = bug_des_com_list[index_bug]
	# 	for index_dev, d in enumerate(dev_id):
	# 		fixer_vector = dev_structure_list[index_dev]
	# 		record = [b, d, np.dot(bug_vector, fixer_vector) / (np.linalg.norm(bug_vector) * np.linalg.norm(fixer_vector))]
	# 		cos_similarity.append(record)
	bug_des_com_list, dev_structure_list = normalize(bug_des_com_list), normalize(dev_structure_list)
	results = bug_des_com_list.dot(dev_structure_list.T)
	results_df = pd.DataFrame(results.flatten())
	bugid_devid = []
	for b in bugid:
		for d in dev_id:
			bugid_devid.append([b, d])
	bugid_devid_df = pd.DataFrame(bugid_devid)
	cos_similarity_df = pd.concat([bugid_devid_df, results_df], axis=1)
	cos_similarity_df.columns = ['bugid', 'fixers', 'cos']
	# cos_similarity_df.to_csv(text_path + filename + "_cos_similarity.csv", index=False)
	# 3.get top3 fixers
	# cos_similarity_df = pd.read_csv(text_path + filename + "_cos_similarity.csv")
	top3_df = cos_similarity_df.groupby('bugid').apply(lambda x:x.sort_values('cos')[-10:])
	return top3_df


def structure():
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_100.csv")
	dev_structure = pd.read_csv(structure_path + filename + "_emb_result_100.csv", header=None)
	# 1.get bugid and dev id
	bugid = list(bug_des_com['bugid'])
	dev_id = list(dev_structure[0])
	# 2.count similarity of one bug and all devs
	bug_des_com.drop(['bugid'], axis=1, inplace=True)
	bug_des_com_list = np.array(bug_des_com)
	dev_structure.drop(0, axis=1, inplace=True)
	dev_structure_list = np.array(dev_structure)
	top3_cos_df = top3_cos_similarity(bug_des_com_list, dev_structure_list, bugid, dev_id)
	top3_bugid_fixer = top3_cos_df.drop(['cos'], axis=1)
	acc = count_fixer_acc(top3_bugid_fixer)


def text_info():
	# 1.concat all vectors of dev
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_132.csv")
	dev_com = pd.read_csv(text_path + filename + "_fixer_component51.csv")
	dev_pro = pd.read_csv(text_path + filename + "_fixer_product51.csv")
	dev_abs = pd.read_csv(text_path + filename + "_fixer_abstract.csv")
	dev_com = dev_com.fillna(0)
	dev_com = dev_com.drop('fixers', axis=1)
	dev_pro = dev_pro.fillna(0)
	dev_pro = dev_pro.drop('fixers', axis=1)
	dev_vec = pd.concat([dev_abs, dev_com, dev_pro], axis=1)
	dev_vec.to_csv(text_path + filename + "_fixer_textInfo.csv", index=False)
	# 2.get bugid and dev id
	bugid = list(bug_des_com['bugid'])
	dev_id = list(dev_vec['fixers'])
	# 3.count similarity of one bug and all devs
	bug_des_com.drop(['bugid'], axis=1, inplace=True)
	bug_des_com_list = np.array(bug_des_com)
	dev_vec.drop(['fixers'], axis=1, inplace=True)
	dev_vec_list = np.array(dev_vec)
	top3_cos_df = top3_cos_similarity(bug_des_com_list, dev_vec_list, bugid, dev_id)
	top3_bugid_fixer = top3_cos_df.drop(['cos'], axis=1)
	acc = count_fixer_acc(top3_bugid_fixer)


def all_info():
	# 1.concat text vectors and structure vectors
	bug_des_com = pd.read_csv(text_path + filename + "_bugid_des_com_232.csv")
	dev_text = pd.read_csv(text_path + filename + "_fixer_textInfo.csv")
	dev_strc = pd.read_csv(structure_path + filename + "_emb_result_100.csv", header=None)
	dev_strc.columns = ['fixers'] + list(range(dev_strc.shape[1] - 1))
	dev_vec = pd.merge(dev_text, dev_strc, how='inner', on='fixers')
	# 2.get bugid and dev id
	bugid = list(bug_des_com['bugid'])
	dev_id = list(dev_vec['fixers'])
	# 3.count similarity of one bug and all devs
	bug_des_com.drop(['bugid'], axis=1, inplace=True)
	bug_des_com_list = np.array(bug_des_com)
	dev_vec.drop(['fixers'], axis=1, inplace=True)
	dev_vec_list = np.array(dev_vec)
	top3_cos_df = top3_cos_similarity(bug_des_com_list, dev_vec_list, bugid, dev_id)
	top3_bugid_fixer = top3_cos_df.drop(['cos'], axis=1)
	acc = count_fixer_acc(top3_bugid_fixer)


def count_fixer_acc(top3_df):
	top1_df = top3_df.drop_duplicates(['bugid'], keep='last')
	top1_tmp_df = pd.merge(bug_tosser_fixer, top1_df, on=['bugid', 'fixers'])
	top1_acc = len(top1_tmp_df) / len(top1_df)
	print(len(top1_df))
	print('top1_acc:', top1_acc)
	top3_tmp_df = pd.merge(bug_tosser_fixer, top3_df, on=['bugid', 'fixers'])
	top3_acc = len(top3_tmp_df) / len(top1_df)
	print('top3_acc:', top3_acc)


def main():
	# 1.get bug-fixer-tosser's dict
	global bug_tosser_fixer
	bug_tosser_fixer = pd.read_csv(fix_path + filename + "_bugid_tossers_fixers.csv")
	# 2.structure - 100
	structure()
	# 3.text - 132
	# text_info()
	# all
	# all_info()


if __name__ == '__main__':
	main()