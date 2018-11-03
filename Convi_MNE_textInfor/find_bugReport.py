# -*- coding: utf-8 -*-
import pandas as pd
import os

def get_eclipse_bugID(dev_pre):
	path = "/media/mia/文档/wubo/eclipse_data/eclipse_history/原始/bughistory_raw/"
	folders = os.listdir(path)
	all_path = []
	for folder in folders:
		all_path.append(path + folder)
	bugID = []
	bug_file = pd.DataFrame()
	for ap in all_path:
		bugs = os.listdir(ap)
		for bug in bugs:
			try:
				bug_file = pd.read_csv(ap + '/' + bug)
				bug_file = bug_file[(bug_file[' What'] == 'CC') | (bug_file[' What'] == 'Assigned')]
				if len(bug_file):
					bug_mail_pre = (set(bug_file[' Who'].apply(lambda x:x.split('@')[0]))) | (set(bug_file[' Added'].apply(lambda x:x.split('@')[0])))
			except Exception, e:
				continue
			if bug_mail_pre & dev_pre:
				bugID.append(bug.split('.')[0])
			else:
				continue
	return bugID

def get_gnome_bugID(dev_pre):
	path = "/media/mia/文档/wubo/Gnome_data/Gnome_History/Gnome_History_select/"
	folders = os.listdir(path)
	bugID = []
	bug_mail_pre = set()
	bug_file = pd.DataFrame()
	count = 0
	for folder in folders:
		try:
			bug_file_csv = pd.read_csv(path + folder)
			bug_file = bug_file_csv[bug_file_csv['What'] == 'Assignee']
			bug_mail_pre = (set(bug_file['Who'].apply(lambda x:x.replace('@', ' ').replace('.', ' ')))) | (set(bug_file['Added'].apply(lambda x:x.replace('@', ' ').replace('.', ' '))))
		except Exception, e:
			count += 1
		if bug_mail_pre & dev_pre:
			bugID.append(folder.split('_')[1].split('.')[0])
		else:
			continue
	# print(count)
	return bugID

def main():
	common_node_path = "/media/mia/文档/paper/eclipse_gnome/multiplex_commonNode/Convi_MNE_textInfor/common_node/"
	common_node_eclipse = "eclipsehistory.csv"
	common_node_gnome = "gnomehistory.csv"
	common_node_eclipse_str = pd.read_csv(common_node_path + common_node_eclipse)
	common_node_gnome_str = pd.read_csv(common_node_path + common_node_gnome)
	dev_pre_eclipse = set(common_node_eclipse_str['From']) | set(common_node_eclipse_str['To'])
	dev_pre_gnome = set(common_node_gnome_str['From']) | set(common_node_gnome_str['To'])

	# eclipse_bugID = get_eclipse_bugID(dev_pre_eclipse)
	# gnome_bugID = get_gnome_bugID(dev_pre_gnome)
	# pd.DataFrame(eclipse_bugID).to_csv("./find_bugReport/eclipse_bugID.csv", index=False, header=None)
	# pd.DataFrame(gnome_bugID).to_csv("./find_bugReport/gnome_bugID.csv", index=False, header=None)



if __name__ == '__main__':
	main()