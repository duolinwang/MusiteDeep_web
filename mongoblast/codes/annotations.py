# OS: Ubuntu, 18.04.1 LTS
# Python: Python 2.7.15
# Mongodb: v3.2.21 
import time
import argparse
import functions
from collections import OrderedDict 

def get_ids(sp):
    ids = []
    table = functions.connectMongoDB('uniprot','table')
    cursor = table.find()
    for doc in cursor:
        if sp !="All":
             if doc['species'] and sp in doc['species']:
                  ids.append(doc['_id'])
        else:
              ids.append(doc['_id']) #save all for -sp=All
    
    return ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp',default="All", help="Species") #before used Metazoa
    #parser.add_argument('-out',default='data_'+ "_".join(time.asctime().split(" ")), help="output folder name")
    parser.add_argument('-out',default='data_uptodate', help="output folder name")
    args = parser.parse_args()
    #ptms =  OrderedDict(
    #        {'Phosphoserine_Phosphothreonine':[],
    #        'Phosphotyrosine':[],
    #        'N-linked (GlcNAc) asparagine':[],
    #        'O-linked (GlcNAc) serine_O-linked (GlcNAc) threonine':[],
    #        'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)':[],
    #        'Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO)':[],
    #        'N6-acetyllysine':[],
    #        'Omega-N-methylarginine_Dimethylated arginine_Symmetric dimethylarginine_Asymmetric dimethylarginine':[],
    #        'N6-methyllysine_N6,N6-dimethyllysine_N6,N6,N6-trimethyllysine':[],
    #        'Pyrrolidone carboxylic acid':[],
    #        'S-palmitoyl cysteine': [],
    #        '3-hydroxyproline_4-hydroxyproline':[],
    #        '4,5-dihydroxylysine_3,4-dihydroxylysine_5-hydroxylysine':[]
    #        })
    #        
    #        
    ptms =  OrderedDict();
    ptms['Phosphoserine_Phosphothreonine']=[]
    ptms['Phosphotyrosine']=[]
    ptms['N-linked (GlcNAc) asparagine']=[]
    ptms['O-linked (GlcNAc) serine_O-linked (GlcNAc) threonine']=[],
    ptms['Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)']=[]
    ptms['Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO)']=[]
    ptms['N6-acetyllysine']=[]
    ptms['Omega-N-methylarginine_Dimethylated arginine_Symmetric dimethylarginine_Asymmetric dimethylarginine']=[]
    ptms['N6-methyllysine_N6,N6-dimethyllysine_N6,N6,N6-trimethyllysine']=[]
    ptms['Pyrrolidone carboxylic acid']=[]
    ptms['S-palmitoyl cysteine']=[]
    ptms['3-hydroxyproline_4-hydroxyproline']=[]
    ptms['4,5-dihydroxylysine_3,4-dihydroxylysine_5-hydroxylysine']=[]
    species = args.sp
    ids = get_ids(species)
    folder_path = args.out
    
    functions.MongotoPTMannotation(ids,ptms,folder_path)
  
if __name__== "__main__":
	main()



