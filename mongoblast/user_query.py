#OS: Ubuntu 16.04.5 LTS
#Python: Python 3.5 
#Mongodb: v3.2.22  

import os
import argparse
import subprocess
ptm_values={
"Phosphotyrosine":"\"Phosphotyrosine\"",
"Phosphoserine":"Phosphoserine",
"Phosphothreonine":"Phosphothreonine",
"N-linked_glycosylation":"\"N-linked (GlcNAc) asparagine\"",
"O-linked_glycosylation":"\"O-linked (GlcNAc) serine_O-linked (GlcNAc) threonine\"",
"Ubiquitination":"\"Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in ubiquitin)\"",
"SUMOylation":"\"Glycyl lysine isopeptide (Lys-Gly)(interchain with G-Cter in SUMO)\"",
"N6-acetyllysine": "\"N6-acetyllysine\"",
"Methylarginine":"\"Omega-N-methylarginine_Dimethylated arginine_Symmetric dimethylarginine_Asymmetric dimethylarginine\"",
"Methyllysine": "\"N6-methyllysine_N6,N6-dimethyllysine_N6,N6,N6-trimethyllysine\"",
"Pyrrolidone_carboxylic_acid":"\"Pyrrolidone carboxylic acid\"",
"S-palmitoyl_cysteine":"\"S-palmitoyl cysteine\"",
"Hydroxyproline":"3-hydroxyproline_4-hydroxyproline",
"Hydroxylysine":"4,5-dihydroxylysine_3,4-dihydroxylysine_5-hydroxylysine"
}

def main():
    path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-query', help="query sequence in FASTA format (currently, only one sequence is allowed).",required=True)
    parser.add_argument('-ptms', default='Phosphoserine;Phosphothreonine', help="PTMs to be annotated (default: 'Phosphoserine;Phosphothreonine'). For multiple ones, use \";\" to seperate them. For example: ptm1;ptm2;ptm3.\n\
                       Currently, we supported\n: \
                       Phosphoserine, Phosphothreonine, Phosphotyrosine,\
                       N-linked_glycosylation,\
                       O-linked_glycosylation,\
                       Ubiquitination,\
                       SUMOylation,\
                       N6-acetyllysine,\
                       Methylarginine,\
                       Methyllysine,\
                       Pyrrolidone_carboxylic_acid,\
                       S-palmitoyl_cysteine,\
                       Hydroxyproline,\
                       Hydroxylysine"
                       )
    
    parser.add_argument('-evalue', default=1e-5, help="blastp evalue (default:1e-5)",required=False)
    parser.add_argument('-max_target_seqs', default=50, help="Maximum number of target sequences to be returned (default:50)",required=False)
    parser.add_argument('-o', default=path+'/display',help="output folder name (default: ./display)",required=False)
    args = parser.parse_args()
    ptms = "_".join([ptm_values[x] for x in args.ptms.split(";")])
    print("selected PTM:"+ptms+"\n")
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
        
    step_1 = 'blastp -query '+query+' -db mydb -evalue '+str(evalue)+' -max_target_seqs '+str(max_target_seqs)+' -outfmt 11 -out '+out_folder+'/format11.asn'
    step_2 = 'python3 '+path+'/codes/blast_parse.py -l '+out_folder+'/format11.asn'+' -ptms '+ptms+' -o '+out_folder
    step_3 = 'python3 '+path+'/codes/merge_results.py -queryId \"'+queryId+'\" -blastFolder '+out_folder+' -ptms '+ptms
    subprocess.call([step_1],shell=True)
    subprocess.call([step_2],shell=True)
    subprocess.call([step_3],shell=True)
    print("User query Finished")


if __name__== "__main__":
	main()

#python3 user_query.py -query example_query_seqs.fasta -ptms "Phosphoserine_Phosphothreonine_Phosphotyrosine"  -o blastoutput
