import pandas as pd
import bs4
import requests
from lxml import html
import re
import time

filename = 'eclipse' # gnome
url = "https://bugs.eclipse.org/bugs/show_bug.cgi?id=" # https://bugzilla.gnome.org/show_bug.cgi?id=


bug_text = pd.DataFrame()
s = requests.Session()
datas = {'username':"674198105@qq.com", 'password':"@120132013lh@"}
head = {"Mozilla/5.0 (X11; Linux x86_64)"}


bugid_path = './assists_dev_MNE/'
bugid_path = pd.read_csv(bugid_path + filename + '_bugid_path.csv')
bugid = list(bugid_path['bugid'])
bug_text['bugid'] = pd.Series(bugid)
product = []
component = []
abstract = []
des_com = []

for bug in bugid:
	print(bug)
	time.sleep(1)
	bug_url = url + str(bug)
	try:
		bug_source = s.post(bug_url, data=datas).text
		soup = bs4.BeautifulSoup(bug_source, "lxml")
		product.append(soup.find_all('td', id='field_container_product')[0].string.strip())
		component.append(list(soup.find_all('td', id='field_container_component')[0].strings)[0].strip('(').strip())
		abstract.append(soup.find_all('span', id='short_desc_nonedit_display')[0].string)
		des_com.append('++'.join(list(map(lambda x:' '.join(list(map(lambda x:x.strip().replace('\n', ''), x.strings))), soup.find_all('pre', class_='bz_comment_text')))))
	except Exception, e:
		print("error", bug)
	
bug_text['product'] = product
bug_text['component'] = component
bug_text['abstract'] = abstract
bug_text['des_com'] = des_com

bug_text.to_csv("./find_bugReport/" + filename + "_bugid_text.csv", index=False)
