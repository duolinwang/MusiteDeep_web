#OS: Ubuntu 16.04.5 LTS
#Python: Python 3.5 
#Mongodb: v3.2.22  

import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-queryId', help="query sequence Id.",required=True)
    parser.add_argument('-blastFolder',help="Blast result folder",required=True)
    parser.add_argument('-ptms', default='Phosphoserine_Phosphothreonine', help="PTMs to be annotated")
    args = parser.parse_args()
    #print(args.ptms)
    queryId=args.queryId
    blastFolder = args.blastFolder
    ptm_list = args.ptms.split("_")
    ptm_results = {}
    for ptm in ptm_list:
        ptmfile= open(os.path.join(blastFolder,ptm+".txt"))
        for line in ptmfile:
            pid = line.split(" ")[0]
            poses = line.strip().split(" ")[1:]
            if pid not in ptm_results.keys():
               ptm_results[pid]={}
               for pos in poses:
                   if pos not in ptm_results[pid].keys():
                      ptm_results[pid][pos]=[]
                      ptm_results[pid][pos].append(ptm)
                   else:
                      ptm_results[pid][pos].append(ptm)
                
            else:
                for pos in poses:
                   if pos not in ptm_results[pid].keys():
                      ptm_results[pid][pos]=[]
                      ptm_results[pid][pos].append(ptm)
                   else:
                      ptm_results[pid][pos].append(ptm)
    
    blast_output = open(os.path.join(blastFolder,"blast_output.txt"))
    output = open(os.path.join(blastFolder,"blastresult.txt"),'w')
    index=0
    for line in blast_output:
        if index==0:
           queryseq = line.strip().split(" ")[2]
           output.write(queryId+"\n")
           output.write(queryseq+"\n")
        else:
           pid = line.split(" ")[0]
           identity = line.split(" ")[1]
           sequence = line.split(" ")[2]
           output.write(pid+"\t("+identity+")"+"\t")
           if pid in ptm_results.keys():
                poses = [int(x) for x in ptm_results[pid].keys()]
                ptmresult = ""
                for pos in sorted(poses):
                    ptmresult+=str(pos)+":"+",".join(ptm_results[pid][str(pos)])+";"
           
           output.write(ptmresult+"\n")
           output.write(sequence)
        
        index+=1


if __name__== "__main__":
	main()