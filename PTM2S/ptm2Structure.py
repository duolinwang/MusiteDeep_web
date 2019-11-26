#OS: Ubuntu 16.04.5 LTS
#Python: Python 3.5 
#Mongodb: v3.2.22  

import os
import argparse
import requests
import requests
from requests.auth import HTTPDigestAuth
import json
from collections import OrderedDict 

def change_fasta2seqs_return(file):
    prot_id = ''
    prot_seq = ''
    seqs = []
    ids = []
    for line in open(file, 'r'):
        if line[0] == '>':
            if prot_id != '':
                seqs.append(prot_seq.upper())
                ids.append(prot_id)
            prot_id = line.strip()
            prot_seq = ''
        elif line.strip() != '':
            prot_seq = prot_seq + line.strip()
    
    if prot_id != '':
        seqs.append(prot_seq.upper())
        ids.append(prot_id)
    
    return ids,seqs

def parse_g2s(jData,positionannotation,maxPDB):
  output = []
  index=0
  for j in jData:
     if maxPDB !='all':
           if index>=int(maxPDB):
              break
     
     tempdir=OrderedDict()
     tempdir['pdbNo'] =j['pdbNo']
     tempdir['pdbId']=j['pdbId']
     tempdir['chain'] = j['chain']
     tempdir['evalue'] = j['evalue']
     tempdir['bitscore'] = j['bitscore']
     tempdir['identity'] = j['identity']
     tempdir['identityPositive']=j['identityPositive']
     tempdir['pdbFrom'] = j['pdbFrom']
     tempdir['pdbTo'] = j['pdbTo']
     tempdir['seqFrom'] = j['seqFrom']
     tempdir['seqTo'] = j['seqTo']
     residueMapping=j['residueMapping']
     PTMannotation=[]
     for resIndex in range(len(residueMapping)):
         pos = str(residueMapping[resIndex]['queryPosition'])
         if pos in positionannotation.keys():
              #item = residueMapping[resIndex]
              #item['PTMscores']=positionannotation[pos]
              item = str(residueMapping[resIndex]['queryPosition'])+":"+residueMapping[resIndex]['queryAminoAcid']+":"+residueMapping[resIndex]['pdbAminoAcid']+":"+positionannotation[pos]
              PTMannotation.append(item)
     
     tempdir['PTMannotation']=PTMannotation
     output.append(tempdir)
     index+=1
  
  return output


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-ptmInput', help="a file contains protein sequences to be predicted in the FASTA format.",required=True)
    parser.add_argument('-ptmOutput', help="the prediction results obtained from MusiteDeep predictor for the input FASTA file. If doesn't have one, please run predict_multi_batch.py first,refer to ../MusiteDeep for details.",required=True)
    parser.add_argument('-o',help="output folder name",required=True)
    parser.add_argument('-maxPDB',default='all',help="max number of returned PDB structrues, (default:'all', return all matched PDB results)",required=False)
    args = parser.parse_args()
    ptmInput = args.ptmInput
    ptmOutput = args.ptmOutput
    out_folder = args.o
    maxPDB=args.maxPDB
    #ptmInput = "./test_seq.fasta"
    #ptmOutput = "Prediction_results.txt"
    #out_folder="./"
    output = open(ptmOutput,'r')
    results = output.readlines()
    outputhash={}
    for j in range(1,len(results)):
        if results[j] == '':
             continue
        if results[j].startswith(">"):
             id = results[j].strip()
             j+=1
        
        pos = results[j].strip().split("\t")[0]
        lastshow = results[j].strip().split("\t")[3]
        if id not in outputhash.keys():
              outputhash[id]=[]
              outputhash[id].append(pos+"\t"+lastshow)
        else:
              outputhash[id].append(pos+"\t"+lastshow)
        
    
    ids,seqs = change_fasta2seqs_return(ptmInput)
    index = 0
    ResultList=[]
    for id in ids:
        print("Processing "+id+"\n")
        Results = OrderedDict()
        Results['ProteinId']=id
        Results['PTM2Structure']=[]
        seq = seqs[index]
        posList = []
        positionannotation={}
        if id in outputhash.keys():
              for j in range(len(outputhash[id])):# all positions for this protein
                    pos = outputhash[id][j].split("\t")[0]
                    score = outputhash[id][j].split("\t")[1]
                    if score != 'None':
                          posList.append(pos)
                          positionannotation[pos]=score
        
        #parse url
        position = "%2C".join(posList)
        url = "https://g2s.genomenexus.org/api/alignments/residueMapping?sequence="+seq+"&positionList="+position;
        myResponse = requests.get(url)
        if(myResponse.ok):
               # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
                   # In this Example, jData are lists of Residues from Genome Mapping to Protein Structures
               jData = json.loads(myResponse.content.decode('utf-8'),object_pairs_hook=OrderedDict)
               Results['PTM2Structure'] = parse_g2s(jData,positionannotation,maxPDB)
        else:
            myResponse.raise_for_status()
            Results['PTM2Structure']=["G2S access error"]
        
        ResultList.append(Results)
        index+=1
    
    with open(os.path.join(out_folder,'ptm2Structure.json'), 'w') as json_file:
         json.dump(ResultList, json_file)
    
    print("Finished")


if __name__== "__main__":
	main()