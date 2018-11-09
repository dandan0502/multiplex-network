import pandas as pd

filename = 'eclipse' # gnome

def preprocess(data):
	pass

def cal_work_theme(l):
	total_bug = int(fixer_bugNum.loc[fixer_bugNum['fixers'] == l['fixers']]['num'])
	l['prob'] = l['num'] / total_bug
	return l


def work_component(bug_tosser_fixer, bug_component):
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
	fixer_com_bugNum = fixer_com_bugNum.apply(cal_work_theme, axis=1)
	
	print(fixer_coms_prob)


def main():
	raw_data_path = "./find_bugReport/"
	assist_dev_path = './assists_dev_MNE/'
	raw_data = pd.read_csv(raw_data_path + filename + "_bugid_text.csv", header=None)
	bug_tosser_fixer = pd.read_csv(assist_dev_path + filename + "_bugid_tossers_fixers.csv")
	raw_data.columns = ['bugid', 'product', 'component', 'abstract', 'des_com']
	bug_component = raw_data.drop(['product', 'abstract', 'des_com'], axis=1)

	preprocess(raw_data)
	work_component(bug_tosser_fixer, bug_component)

if __name__ == '__main__':
	main()