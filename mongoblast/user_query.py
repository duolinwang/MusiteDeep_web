#OS: Ubuntu 16.04.5 LTS
#Python: Python 3.5 
#Mongodb: v3.2.22  

import os
import argparse
import subprocess

def main():
    path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-query', help="query sequence in FASTA format (currently, only one sequence is allowed).",required=True)
    parser.add_argument('-l', default=path,help="local filepath, where to put the intermediate blast results (default: current folder)",required=False)
    parser.add_argument('-ptms', default='Phosphoserine_Phosphothreonine', help="PTMs to be annotated (default: 'Phosphoserine_Phosphothreonine'). For multiple ones, use \"_\" to seperate them. For example: ptm1_ptm2_ptm3.\n\
                       Currently, we supported\n: Phosphoserine, Phosphothreonine, Phosphotyrosine,\
                       N-linked (GlcNAc) asparagine,\
                       O-linked (GlcNAc) serine_O-linked (GlcNAc) threonine,\
                       Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin),\
                       Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO),\
                       N6-acetyllysine,\
                       Omega-N-methylarginine_ Dimethylated arginine_Symmetric dimethylarginine_Asymmetric dimethylarginine,\
                       N6-methyllysine_N6,N6-dimethyllysine_N6,N6,N6-trimethyllysine,\
                       Pyrrolidone carboxylic acid,\
                       S-palmitoyl cysteine,\
                       3-hydroxyproline_4-hydroxyproline,\
                       4,5-dihydroxylysine_3,4-dihydroxylysine_5-hydroxylysine"
                       )
    
    parser.add_argument('-evalue', default=1e-5, help="blastp evalue (default:1e-5)",required=False)
    parser.add_argument('-max_target_seqs', default=50, help="Maximum number of target sequences to be returned (default:50)",required=False)
    parser.add_argument('-o', default=path+'/display',help="output folder name (default: ./display)",required=False)
    args = parser.parse_args()
    filepath = args.l
    ptms = args.ptms
    evalue = args.evalue
    out_folder = args.o
    max_target_seqs = args.max_target_seqs
    query=args.query
    fd = open(query)
    count=0
    for line in fd:
        if line.startswith('>'):
            queryId=line.strip()
            count+=1
    
    if count>1:
        print("More than one sequences are provided, but we currently accept only one query sequence!")
        exit(1)
        
    step_1 = 'blastp -query '+query+' -db mydb -evalue '+str(evalue)+' -max_target_seqs '+str(max_target_seqs)+' -outfmt 11 -out '+filepath+'/format11.asn'
    step_2 = 'python3 codes/blast_parse.py -l '+filepath+'/format11.asn'+' -ptms '+ptms+' -o '+out_folder
    step_3 = 'python3 codes/merge_results.py -queryId \"'+queryId+'\" -blastFolder '+out_folder+' -ptms '+ptms
    subprocess.call([step_1],shell=True)
    subprocess.call([step_2],shell=True)
    subprocess.call([step_3],shell=True)
    print("User query Finished")


if __name__== "__main__":
	main()