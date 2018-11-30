import pandas as pd
import numpy as np
from collections import Counter
from fuzzywuzzy import fuzz, process
import lda
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

filename = 'gnome' # gnome

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

def clean_text(l):
	'''
	去标点、数字、分割成单词、去掉停词、词干提取
	'''
	l = re.sub('[^a-zA-Z]', ' ', str(l))
	l = nltk.word_tokenize(l)
	stop_word = stopwords.words('english') 
	del_stop = [w for w in l if w not in stop_word and wordnet.synsets(w)]
	clean_word = []
	for w in del_stop:
		word = WordNetLemmatizer().lemmatize(w) # 词性还原
		word = PorterStemmer().stem(word) # 词干提取
		clean_word.append(word)
	l = clean_word
	return l


def preprocess(data, bug_tosser_fixer):
	data.drop_duplicates(['bugid'], keep='last', inplace=True)
	data.drop(['product', 'component'], axis=1, inplace=True)
	# process abstract and des_com
	delEStr = string.punctuation + string.digits # 去掉Ascii标点符号和数字
	data['abstract'] = data['abstract'].map(clean_text)
	data['des_com'] = data['des_com'].map(clean_text)
	# group fixers' abstracts
	fixer_data = pd.merge(data, bug_tosser_fixer)
	fixer_group_abstract = fixer_data['abstract'].groupby(fixer_data['fixers']).sum().reset_index()
	fixer_group_abstract.columns = ['fixers', 'abstract']
	data.drop(['abstract'], axis=1, inplace=True)
	data = data.reset_index(drop=True)
	return data, fixer_group_abstract


def topic_extraction(corpus, ntopics):
	# gensim lda
	common_dictionary = Dictionary(corpus)
	common_corpus = [common_dictionary.doc2bow(text) for text in corpus]
	lda = LdaModel(common_corpus, num_topics=ntopics, iterations=800, random_state=1)
	features = lda.get_document_topics(common_corpus, minimum_probability=0)
	lda_list = []
	for f in features:
		lda_list.append([b[1] for b in f])
	lda_df = pd.DataFrame(lda_list)
	lda_df = lda_df.reset_index(drop=True)
	return lda_df


def cal_work_theme(l):
	total_bug = int(fixer_bugNum.loc[fixer_bugNum['fixers'] == l['fixers']]['num'])
	l['prob'] = l['num'] / total_bug
	return l


def trans_component(l):
	if l['component'] in top_component:
		return l
	elif process.extractOne(l['component'], top_component)[1] >= 90:
		l['component'] = process.extractOne(l['component'], top_component)[0]
	else:
		l['component'] = 'others'
	return l

# work component + work product
def work_component(bug_tosser_fixer, bug_component, n):
	bug_component.columns = ['bugid', 'component']
	bug_fixer = bug_tosser_fixer.drop(['tossers'], axis=1)
	fixer = set(bug_fixer['fixers'])
	work_com = pd.DataFrame()
	fixer_df = bug_fixer.loc[bug_fixer['fixers'].isin(fixer)] # all fixers and bugs
	fixer_bugid_component = pd.merge(fixer_df, bug_component)
	global fixer_bugNum 
	fixer_bugNum = fixer_bugid_component['bugid'].groupby(fixer_bugid_component['fixers']).count().reset_index()
	fixer_bugNum.columns = ['fixers', 'num']
	fixer_com_bugNum = fixer_bugid_component['bugid'].groupby([fixer_bugid_component['fixers'], fixer_bugid_component['component']]).count().reset_index()
	fixer_com_bugNum.columns = ['fixers', 'component', 'num']
	fixer_com_prob = fixer_com_bugNum.apply(cal_work_theme, axis=1) # fixer,xomponent,prob
	# trans component to vectors
	all_component = list(fixer_com_prob['component'])
	global top_component
	top_component = [x[0] for x in sorted(Counter(all_component).items(), key=lambda x: x[1], reverse=True)][:n]
	other_component = list(set(all_component) - set(top_component))
	fixer_trans_com_prob = fixer_com_prob.apply(trans_component, axis=1)
	fixer_trans_com_prob.drop_duplicates(['fixers', 'component'], keep='first', inplace=True)
	return fixer_trans_com_prob.pivot('fixers', 'component', 'prob').reset_index()


def pre_raw_data(data):
	# process abstract and des_com
	delEStr = string.punctuation + string.digits # 去掉Ascii标点符号和数字
	data['abstract'] = data['abstract'].map(clean_text)
	data['des_com'] = data['des_com'].map(clean_text)
	return data


def sum_all_text(l):
	bugs = l['bugids'].split('++')
	sel_text = all_text_raw_data.loc[all_text_raw_data['bugid'].isin(bugs)]
	sel_text_list = sel_text.drop(['bugid'], axis=1)
	sel_text_list = sel_text_list.values.tolist()
	sel_text_list = list(map(lambda x:str(x[0]), sel_text_list))
	l['all_text'] = sel_text_list
	# print(sel_text_list)
	return l


def all_info(raw_data, tossers_bug_df):
	filter_raw_data = pre_raw_data(raw_data)
	filter_raw_data['des_com'] = filter_raw_data['des_com'] + filter_raw_data['abstract']
	# print(filter_raw_data['des_com'])
	global all_text_raw_data
	all_text_raw_data = filter_raw_data.drop(['abstract'], axis=1)
	tossers_bug_all_text = tossers_bug_df.apply(sum_all_text, axis=1)
	return tossers_bug_all_text


def sum_bug_text(l):
	l['all_text'] = [l['product']] + [l['component']] + l['des_com']
	return l


def bugs_all_info(raw_data):
	filter_raw_data = pre_raw_data(raw_data)
	filter_raw_data['des_com'] = filter_raw_data['des_com'] + filter_raw_data['abstract']
	filter_raw_data.drop(['abstract'], axis=1, inplace=True)

	global all_text_raw_data
	all_text_raw_data = filter_raw_data.apply(sum_bug_text, axis=1)
	return all_text_raw_data


def main():
	raw_data_path = "./find_bugReport/"
	assist_dev_path = './assists_dev_MNE/'
	# ----------------------------------------------------------------------------------
	# the raw_data file name has been changed because try the data without comments
	# raw_data = pd.read_csv(raw_data_path + filename + "_bugid_text.csv", header=None, lineterminator="\n")
	raw_data = pd.read_csv(raw_data_path + filename + "_bugid_text_del_com.csv", header=None, lineterminator="\n")
	# ----------------------------------------------------------------------------------
	bug_tosser_fixer = pd.read_csv(assist_dev_path + filename + "_bugid_tossers_fixers.csv")
	raw_data.columns = ['bugid', 'product', 'component', 'abstract', 'des_com']
	raw_data['bugid'] = raw_data['bugid'].astype('str')
	bug_component = raw_data.drop(['product', 'abstract', 'des_com'], axis=1)
	bug_product = raw_data.drop(['component', 'abstract', 'des_com'], axis=1)

	# 1.clean des_com, add all abstracts and clean them
	# bugid_des_com, fixer_abstract = preprocess(raw_data, bug_tosser_fixer)
	# print(bugid_des_com, fixer_abstract)

	# 2.get bugid des_com 232/132/100/200
	# bugid_des_com_df_232 = pd.concat([bugid_des_com['bugid'], topic_extraction(list(bugid_des_com['des_com']), 232)], axis=1)
	# bugid_des_com_df_232.to_csv("./textInfo/" + filename + "_bugid_des_com_232.csv", index=None)
	# print("finish bugid des_com 232")

	# bugid_des_com_df_132 = pd.concat([bugid_des_com['bugid'], topic_extraction(list(bugid_des_com['des_com']), 132)], axis=1)
	# bugid_des_com_df_132.to_csv("./textInfo/" + filename + "_bugid_des_com_132.csv", index=None)
	# print("finish bugid des_com 132")

	# topics = topic_extraction(list(bugid_des_com['des_com']), 100)
	# bugid_des_com_df_100 = pd.concat([bugid_des_com['bugid'], topics], axis=1)
	# bugid_des_com_df_100.to_csv("./textInfo/" + filename + "_bugid_des_com_100.csv", index=None)
	# print("finish bugid des_com 100")

	# bugid_des_com_df_68 = pd.concat([bugid_des_com['bugid'], topic_extraction(list(bugid_des_com['des_com']), 68)], axis=1)
	# bugid_des_com_df_68.to_csv("./textInfo/" + filename + "_bugid_des_com_68.csv", index=None)
	# print("finish bugid des_com 68")

	# 3.get fixer abstract
	# fixer_abstract_df = pd.concat([fixer_abstract['fixers'], topic_extraction(list(fixer_abstract['abstract']), 248)], axis=1)
	# fixer_abstract_df.to_csv("./textInfo/" + filename + "_fixer_abstract_248.csv", index=None)
	# print("finish fixer abstract")

	# 4.get 51 dimensions vectcors with component and product
	# fixer_component51 = work_component(bug_tosser_fixer, bug_component, 150)
	# fixer_component51.to_csv("./textInfo/" + filename + "_fixer_component151.csv", index=None)
	# print("finish fixer component")
	# fixer_product51 = work_component(bug_tosser_fixer, bug_product, 150)
	# fixer_product51.to_csv("./textInfo/" + filename + "_fixer_product151.csv", index=None)
	# print("finish fixer product")

	# 5.use the same semantic space to get vectors of devs
	# ----------------get a dataframe of tossers and bugs, run it only once-----------------
	# bug_tosser_fixer_list = bug_tosser_fixer.values.tolist()
	# tossers_bug_dict = {}
	# for bug in bug_tosser_fixer_list:
	# 	tossers = bug[1].split('++')
	# 	for t in tossers:
	# 		if t not in tossers_bug_dict:
	# 			tossers_bug_dict[t] = str(bug[0])
	# 		else:
	# 			tossers_bug_dict[t] = str(tossers_bug_dict[t]) + '++' + str(bug[0])
	# tossers_bug_df = pd.DataFrame.from_dict(tossers_bug_dict, orient='index').reset_index()
	# tossers_bug_df.columns = ['tossers', 'bugids']
	# tossers_bug_df.to_csv(assist_dev_path + filename + '_tossers_bugids.csv', index=None)
	# -----------------------------------------------------------------------------------
	tossers_bug_df = pd.read_csv(assist_dev_path + filename + '_tossers_bugids.csv')
	tossers_bug_all_text = all_info(raw_data, tossers_bug_df)
	vector_num = [100, 200, 400]
	for v in vector_num:
		tossers_bug_all_text_topics = pd.concat([tossers_bug_all_text['tossers'], topic_extraction(list(tossers_bug_all_text['all_text']), v)], axis=1)
		tossers_bug_all_text_topics.to_csv("./textInfo_delCom/" + filename + "_tossers_bug_all_text_topics_{}.csv".format(v), index=None)
	print('tosser finish')
	# 6.use the same semantic space to get vectors of bugs
	bugs_all_text = bugs_all_info(raw_data)
	#coding by laoge
	bugs_corpus = list(bugs_all_text['all_text'])
	new_bug_corpus = [[str(i) for i in j] for j in bugs_corpus]
	#mei le
	vector_num = [100, 200]
	for v in vector_num:
		bugs_all_text_topics = pd.concat([bugs_all_text['bugid'], topic_extraction(new_bug_corpus, v)], axis=1)
		bugs_all_text_topics.to_csv("./textInfo_delCom/" + filename + "_bugs_all_text_topics_{}.csv".format(v), index=None)


if __name__ == '__main__':
	main()