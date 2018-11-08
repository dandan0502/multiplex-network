import pandas as pd

filename = 'eclipse' # gnome

def preprocess(data):
	pass


def work_component(bug_tosser_fixer, bug_component):
	bug_fixer = bug_tosser_fixer.drop(['tossers'], axis=1)
	fixer = set(bug_fixer['fixers'])
	work_com = pd.DataFrame()
	fixer_df = bug_fixer.loc[bug_fixer['fixers'].isin(fixer)] # all fixers and bugs
	fixer_bugid_component = pd.merge(fixer_df, bug_component)
	fixer_bugNum = fixer_bugid_component['bugid'].groupby(fixer_bugid_component['fixers']).count().reset_index()
	fixer_bugNum.columns = ['fixers', 'num']
	fixer_com_bugNum = fixer_bugid_component['bugid'].groupby([fixer_bugid_component['fixers'], fixer_bugid_component['component']]).count().reset_index()
	fixer_com_bugNum.columns = ['fixers', 'component', 'num']


def main():
	raw_data_path = "./find_bugReport/"
	assist_dev_path = './assists_dev_MNE/'
	raw_data = pd.read_csv(raw_data_path + filename + "_bugid_text.csv")
	bug_tosser_fixer = pd.read_csv(assist_dev_path + filename + "_bugid_tossers_fixers.csv")
	bug_component = raw_data.drop(['product', 'abstract', 'des_com'], axis=1)

	preprocess(raw_data)
	work_component(bug_tosser_fixer, bug_component)

if __name__ == '__main__':
	main()