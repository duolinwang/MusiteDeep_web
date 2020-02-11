# OS: Ubuntu, 18.04.1 LTS
# Python: Python 2.7.15
# Mongodb: v3.2.21 
# Siteng Cai
import sys
import os.path
import argparse
import re
import itertools
import functions
 
def is_number(s):
	try:
		int(s)
		return True
	except ValueError:
		pass
	return False

def seq_read(fp):
	line = fp.readline().replace(" ", "").rstrip()
	seq = ""
	while line != '//':
		seq += line
		line = fp.readline().replace(" ", "").rstrip()
	return seq

def tableGeneration(filepath,ptms):
    table = functions.connectMongoDB('uniprot','table')
    table.drop()
    out_id = ""
    out_ac = []
    out_position = []
    out_data = dict()
    sequence = ""
    temp_ptm = ""
    prev_fp_pos = 0
    check = []
    
    fp = open(filepath)
    line = fp.readline()
    
    while line:
        collapsed = ' '.join(line.split())
        data = collapsed.split(";")
        info = data[0].split(" ")
        tag = info[0]
        #print(data)
        #print(info)
        #print(info[0]+" info1 "+info[1]+"\n")
        if tag == "ID":
            out_id = info[1]
        elif tag == "AC":
            out_ac.append(info[1])
            if len(data)  > 2:
                for x in range(1, len(data)-1):
                    out_ac.append(data[x].lstrip())
        elif tag == "OC":
            check.append(info[1].lstrip())
            if len(data) > 2:
                for x in range(1, len(data)-1):
                    check.append(data[x].lstrip())
            out_data = {"_id" : out_id,"ac":out_ac,"species":check}
        elif tag == "FT":
            temp_ptm = ""
            out_position = info[2]
            #temp_ptm = " ".join(info[4:])
            if "P68250" in out_ac:
                print(info)
            
            prev_fp_pos = fp.tell()
            line = ' '.join(fp.readline().split())
            info = line.split(" ")
            while info[0] == "FT":
                if len(info) ==3 and is_number(info[2]):
                    if "P68250" in out_ac:
                        print("###########temp_ptm is 2 "+temp_ptm+"\n")
                        print("out position is "+str(out_position)+"\n")
                    temp_ptm = temp_ptm.partition('/note="')[2].partition('"')[0]
                    temp_ptm = re.sub('(\.*)\)',')',temp_ptm)
                    #if "P0C9J5" in out_ac:
                    #	print("################temp_ptm is 2 "+temp_ptm+"\n")
                    for doc in ptms:
                        #if "Q9TT90" in out_ac and doc == 'Glycyllysineisopeptide(Lys-Gly)(interchainwithG-CterinSUMO)':
                        #	print(doc+" vs "+re.sub('[\.|\;].*','',temp_ptm)+"\n")
                        #if "P0C9J5" in out_ac and doc == 'N-linked (GlcNAc) asparagine':
                        #	print(doc+" 2vs "+re.sub('[\.|\;].*','',temp_ptm)+"\n")
                        
                        if doc == re.sub('[\.|\;].*','',temp_ptm):
                            #if "P0C9J5" in out_ac:
                            #	print("2 yes\n"+"position"+str(out_position)+"\n")
                            ptms.setdefault(doc, []).append(out_position)
                        if doc == 'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)':
                            if 'Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in ubiquitin)' == re.sub('[\.|\;].*','',temp_ptm):
                                ptms.setdefault(doc, []).append(out_position)
                        
                        if doc == 'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO)':
                            if 'Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in SUMO)' == re.sub('[\.|\;].*','',temp_ptm):
                                ptms.setdefault(doc, []).append(out_position)
                    
                    if "P68250" in out_ac:
                        print("###########temp_ptm is 2 "+temp_ptm+"\n")
                        print("out position is "+str(out_position)+"\n")
                    temp_ptm = ""
                    out_position = info[2]
                    #temp_ptm = " ".join(info[4:])
                else:
                    #if not info[1].startswith('/'):
                    temp_ptm = temp_ptm +" "+" ".join(info[1:])
                    #if "Q0P5A7" in out_ac:
                    #   print("#################temp_ptm is 3 "+temp_ptm+"\n")
                    #for i in range(1,len(info)):
                    #	temp_ptm += info[i].rstrip()
                    #print(temp_ptm+"\n")
                
                prev_fp_pos = fp.tell()
                line = ' '.join(fp.readline().split())
                info = line.split(" ")
            temp_ptm = temp_ptm.partition('/note="')[2].partition('"')[0]
            temp_ptm = re.sub('(\.*)\)',')',temp_ptm)
            for doc in ptms:
                #if "Q0P5A7" in out_ac and doc == 'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)':
                #	print(doc+" "+re.sub('[\.|\;].*','',temp_ptm)+"\n")
                
                if doc == re.sub('[\.|\;].*','',temp_ptm):
                    ptms.setdefault(doc, []).append(out_position)
                if doc == 'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)':
                    if 'Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in ubiquitin)' == re.sub('[\.|\;].*','',temp_ptm):
                       #print("yes!")
                       ptms.setdefault(doc, []).append(out_position)
                
                if doc == 'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO)':
                    if 'Glycyl lysine isopeptide (Lys-Gly) (interchain with G-Cter in SUMO)' == re.sub('[\.|\;].*','',temp_ptm):
                        ptms.setdefault(doc, []).append(out_position)
            
            if "P68250" in out_ac:
                        print("ptms1\n")
                        print(ptms)
            
            ptms = dict( [k,v] for k,v in ptms.items() if len(v)>0)
            #ptms = dict( [(k,list(itertools.chain.from_iterable(v))) for k,v in ptms.items() if len(v)>0])
            if "P68250" in out_ac:
                        print("ptms2\n")
                        print(ptms)
            
            fp.seek(prev_fp_pos)
        elif tag == "SQ":
            sequence = seq_read(fp)
            out_data = functions.merge_two_dicts(out_data,ptms)
            out_data['sequence'] = sequence
            table.save(out_data)
            ##rewind
            ptms = {'Phosphoserine':[],'Phosphothreonine':[],
                'Phosphotyrosine':[],
                'N-linked (GlcNAc) asparagine':[],
                'O-linked (GlcNAc) serine':[],'O-linked (GlcNAc) threonine':[],
                'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)':[],
                'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO)':[],
                'N6-acetyllysine':[],
                'Omega-N-methylarginine':[],'Dimethylated arginine':[],'Symmetric dimethylarginine':[],'Asymmetric dimethylarginine':[],
                'N6-methyllysine':[],'N6,N6-dimethyllysine':[],'N6,N6,N6-trimethyllysine':[],
                'Pyrrolidone carboxylic acid':[],
                'S-palmitoyl cysteine': [],
                '3-hydroxyproline':[],'4-hydroxyproline':[],#Hydroxylation P
                '4,5-dihydroxylysine':[], '3,4-dihydroxylysine':[],'5-hydroxylysine':[] #Hydroxylation K
                }
            out_data.clear()
            out_ac = []
            out_position = []
            sequence = ""
            check = []
        
        line = fp.readline()
    
    fp.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', default='uniprotData/uniprot.txt',help="local filepath,default path can trigger auto download")
	parser.add_argument('-update', type=int, default=0, help="update options: check every # months, default to manual(0)")
	parser.add_argument('-download', type=int, default=0, help="whether to download uniprotData/uniprot.txt")
	args = parser.parse_args()
	filepath = args.l
	
	ptms = {'Phosphoserine':[],'Phosphothreonine':[],
			'Phosphotyrosine':[],
			'N-linked (GlcNAc) asparagine':[],
			'O-linked (GlcNAc) serine':[],'O-linked (GlcNAc) threonine':[],
			'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)':[],
			'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO)':[],
			'N6-acetyllysine':[],
			'Omega-N-methylarginine':[],'Dimethylated arginine':[],'Symmetric dimethylarginine':[],'Asymmetric dimethylarginine':[],
			'N6-methyllysine':[],'N6,N6-dimethyllysine':[],'N6,N6,N6-trimethyllysine':[],
			'Pyrrolidone carboxylic acid':[],
			'S-palmitoyl cysteine': [],
			'3-hydroxyproline':[],'4-hydroxyproline':[],#Hydroxylation P
			'4,5-dihydroxylysine':[], '3,4-dihydroxylysine':[],'5-hydroxylysine':[] #Hydroxylation K
			}
	
	if not os.path.exists("uniprotData"):
		os.makedirs("uniprotData")
	
	if args.download >0:
		if filepath == 'uniprotData/uniprot.txt':
			functions.getUniprot()
	
	if os.path.exists(filepath):
		tableGeneration(filepath,ptms)
		if args.update > 0:
			table_date = functions.rssread()
			functions.setAutoUpdate(args.update)
			print("Check for update every %s months!" % (args.update))
			functions.Config_edit(table_date)
	else:
		print("File does not exist\n")
		sys.exit()
  
if __name__== "__main__":
	main()



