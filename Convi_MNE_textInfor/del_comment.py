import pandas as pd

filename = 'gnome' # gnome
raw_data_path = './find_bugReport/'

raw_data = pd.read_csv(raw_data_path + filename + '_bugid_text.csv', header=None, lineterminator="\n")
raw_data.columns = ['bugid', 'product', 'component', 'abstract', 'des_com']
print(raw_data['des_com'])
raw_data['des_com'] = raw_data['des_com'].astype('str')
raw_data['des_com'] = raw_data['des_com'].map(lambda x: x.split('++')[0])
raw_data.to_csv(raw_data_path + filename + '_bugid_text_del_com.csv', index=False, header=None)