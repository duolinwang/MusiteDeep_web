# OS: Ubuntu, 18.04.1 LTS
# Python: Python 2.7.15
# Mongodb: v3.2.21 
# Siteng Cai
import feedparser
from datetime import datetime as dt
import sys
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve
    
import gzip
import shutil
import pymongo
from pymongo import MongoClient
import sys
import os.path
import argparse
from crontab import CronTab
import configparser
import re
import json
import itertools
import os
import getpass
import time
#GLOBAL VAR
USER_NAME = getpass.getuser()
PARENT_DIR = os.path.dirname(os.getcwd())


def rssread():
	url = 'https://www.uniprot.org/news/?format=rss'

	feed = feedparser.parse(url )

	date = feed['updated'].split(' ')
	new_date = date[1]+'/'+date[2]+'/'+date[3]

	new_date = dt.strptime(new_date,"%d/%b/%Y").strftime('%Y-%m-%d')
	return new_date

def getUniprot():
    creattime = '-'.join(time.ctime(os.path.getctime('./uniprotData/uniprot.txt')).split())
    print("zip old file created on "+creattime)
    with gzip.open('./uniprotData/uniprot'+str(creattime)+'.txt.gz', 'wb') as f_out:
        with open('./uniprotData/uniprot.txt', 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
    
    print("download new file")
    uniprot_url = 'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'
    urlretrieve(uniprot_url, 'uniprot.txt.gz')
    print("Unzip file...")
    with gzip.open('uniprot.txt.gz', 'rb') as f_in:
        with open('./uniprotData/uniprot.txt', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("File name:uniprot.txt")
    

def valid_date(s):
    try:
        return dt.strptime(s, "%d/%m/%Y")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def connectMongoDB(dbname,colname):
	# connect to mongodb
	client = MongoClient('localhost', 23333)
	# Get the database
	db = client[dbname]
	collection = db[colname]
	return collection
	
def updateMongoDB(filepath,dbname,colname,date):
	train = 1 
	if date == "1/1/1111":
		train = 0
	try:
		old_date = dt.strptime(date, "%d/%m/%Y")
	except ValueError:
		print("Invalid date!")
		sys.exit()
	collection = connectMongoDB(dbname,colname)
	# Open a file
	out_date = dt.strptime("1/1/1111", "%d/%m/%Y")
	id_flag = 0
	ac_flag = 0
	seq_flag = 0
	out_ac = []
	sequence = ''
	out_data = dict()
	train_ids = []
	

	with open(filepath) as fp:
		for line in fp:
			collapsed = ' '.join(line.split())
			data = collapsed.split(";")
			parsed_1 = data[0].split(" ")
			if seq_flag == 0 and parsed_1[0] == "ID" and  id_flag == 0:
				id_flag = 1
				out_id = parsed_1[1]
			elif seq_flag == 0 and parsed_1[0] == "AC" and  ac_flag == 0:
				ac_flag = 1	
				out_ac.append(parsed_1[1])
				if len(data)  > 2:
					for x in range(1, len(data)-1):
						out_ac.append(data[x])
				out_data = {'_id' : out_id,'ac':out_ac}
			elif seq_flag == 0 and parsed_1[0] == "DT":
				temp_date = dt.strptime(re.sub('[,]', '',parsed_1[1]), "%d-%b-%Y")
				if temp_date > out_date:
					out_date = temp_date
			elif (len(parsed_1[0]) > 2 or seq_flag == 1) and parsed_1[0] != '//':
				seq_flag = 1
				sequence += collapsed
			elif parsed_1[0] == '//':
				
				sequence = ''.join(sequence.split())
				out_data['sequence'] = sequence
				out_data['date'] = out_date
				collection.save(out_data)
				
				if train == 1 and old_date <= out_date:
					train_ids.append(out_id)
				# rewind
				seq_flag = 0
				id_flag = 0
				ac_flag = 0
				out_ac = []
				sequence = ''
				out_date = dt.strptime("1/1/1111", "%d/%m/%Y")
	fp.close()
	update_size = len(train_ids)
	update_date = date
	update_info = {'Database name':dbname,'Collection name':colname,'Number of Entries updated':update_size,'Update date':update_date} 
	if train == 1:
		ids_file = open("./output/train_ids.txt","w")
		for id in train_ids:
			ids_file.write(id)
		ids_file.close()
		info_file = open("./output/info.txt","w")
		json.dump(update_info, info_file)
		info_file.close()


# crontab job scheduler		
def setAutoUpdate(update):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	my_cron = CronTab(USER_NAME)
	cmd = "/usr/bin/python "+dir_path+"/rssReader.py"
	job = my_cron.new(command=cmd)
	job.every(update).months()
	my_cron.write()
	for job in my_cron:
		print (job)
		
def Config_edit(date):
	config = configparser.ConfigParser()
	config['DEFAULT'] = {'date': date}
	with open('config.ini', 'w') as configfile:
		config.write(configfile)

# remove duplicate numbers from list
def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

# merge two python dict two one dict
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

	
#insert # after ptm position in seq, format: fasta
def prepare(id,ft,seq):
	seq_list = list(seq)
	for c,i in enumerate(ft):
		seq_list.insert(i+c,'#')
	
	sequence = ''.join(seq_list)
	out_data = '>sp|'+id+'\n'+sequence+'\n'
	return out_data
#ptm annotation
def MongotoPTMannotation(proteinIDs,Tag_FTs,output_prefix):
    #Tag_FTs are PTMs
    import time
    table = connectMongoDB('uniprot','table')
    
    file = []
    out_data = ''
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
    
    displayfile = open(output_prefix+'/display.txt','w') #for display in PTMweb download ptm
    curtime=time.asctime()
    curtime = ' '.join(curtime.split())
    ptm_proteins=[]
    ptm_sites=[]
    #for index, tag in enumerate(sorted(Tag_FTs)):
    for index, tag in enumerate(Tag_FTs):
        file.append(open(output_prefix+'/'+tag+'.fasta','w'))
        ptm_proteins.append(0) #initializ all the ptm numof proteins and sites with 0
        ptm_sites.append(0)
    
    for id in proteinIDs:
        ptm = table.find_one({'_id': id})
        
        
        for index, ft in enumerate(Tag_FTs):
            ft_index = []
            unfold_ft = ft.split("_")
            
            for new_ft in unfold_ft:
                if new_ft in ptm: #one ptm type in one mongodb item
                    ft_index.extend(ptm[new_ft]) 
            ft_index = map(int,ft_index)
            #ft_index.sort()
            ft_index=list(sorted(set(ft_index)))
            if len(ft_index) >= 1: #this protein has this ptm annotation
                sequence = ptm['sequence']
                ptm_proteins[index]+=1
                ptm_sites[index]+=len(ft_index)
                out_data = prepare(ptm['ac'][0]+"|"+ptm['_id'],ft_index,sequence)
                
                file[index].write(out_data)
    
    displayfile.write(curtime+"\n")
    
    for index,ft in enumerate(Tag_FTs):
        displayfile.write(ft+".fasta"+"\t"+str(ptm_proteins[index])+"\t"+str(ptm_sites[index])+"\n")
    
    displayfile.close()
    for index, tag in enumerate(Tag_FTs):
        file[index].close()
        
