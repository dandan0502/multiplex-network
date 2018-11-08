import pandas as pd

eclipse = "eclipse"
gnome = "gnome"
file_name = eclipse # eclipse,gnome
aaaa = '27' # 27,10
path = "./assists_dev_MNE/"
name_index = pd.read_csv(path + file_name + "_name_index.csv")
name_index["index"] = name_index['index'].astype('str')
name_index_dict = dict(zip(name_index['name'], name_index['index']))
	
def trans_tosser_index(x):
	tosser_list = []
	for i in x:
		try:
			tmp_i = name_index_dict[i]
		except Exception, e:
			tmp_i = aaaa	
		tosser_list.append(tmp_i)
	tosser_list_drop_dup = reduce(lambda x,y: x if y in x else x + [y], [[],] + tosser_list)
	tosser_str = "++".join(tosser_list_drop_dup)
	return tosser_str


def trans_fixer_index(x):
	try:
		x = name_index_dict[x]
	except Exception, e:
		x = aaaa
	return x


def fill_tossers(l):
	if l["tossers"] == []:
		l["tossers"] = l["fixers"]
	return l


def find_tp(path, file_name):
	bugid_path = pd.read_csv(path + file_name + "_bugid_path.csv")
	
	bugid_tossers_fixers = pd.DataFrame()
	tossers = list()
	fixers  = list()
	bugid_path_list = bugid_path.values.tolist()
	for row in bugid_path_list:
		path_split = reduce(lambda x,y: x if y in x else x + [y], [[],] + row[1].split("++"))
		tossers.append(path_split[:-1])
		fixers.append(path_split[-1])
	bugid_tossers_fixers["bugid"] = bugid_path["bugid"]
	bugid_tossers_fixers["tossers"] = pd.Series(tossers)
	bugid_tossers_fixers["fixers"] = pd.Series(fixers)
	bugid_tossers_fixers = bugid_tossers_fixers.apply(fill_tossers, axis=1)
	bugid_tossers_fixers["tossers"] = bugid_tossers_fixers["tossers"].map(trans_tosser_index)
	bugid_tossers_fixers["fixers"] = bugid_tossers_fixers["fixers"].map(trans_fixer_index)
	return bugid_tossers_fixers


def main():
	bugid_tossers_fixers = find_tp(path, file_name)
	bugid_tossers_fixers.to_csv(path + file_name + "_bugid_tossers_fixers.csv", index=False)


if __name__ == '__main__':
 	main()