#
#!/usr/bin/python
#vm: amazon linux 2 AMI
#python 2.7.5
#mongodb 3.6.3
import pymongo
from pymongo import MongoClient
import sys
import os.path
import argparse
import re
import itertools
import functions
	
	
#if there is ft add 1 after id, format: fasta
def prepareData(id,seq):
	out_data = '>sp|'+id+'\n'+seq+'\n'
	return out_data

#convert uniprot DB to fasta file
def	db_to_fasta(output_prefix):
	entry = functions.connectMongoDB('uniprot','table')
	out_data = ''
	out_file = open(output_prefix+'.fasta','w')
	entrys = entry.find({})

	for doc in entrys:			
		out_data = prepareData(doc['_id'],doc['sequence'])
		out_file.write(out_data)

	out_file.close()
		
#requirement: 1. tableGenerator.py 
#example DBtoF.py -l 'background_seqs'
#output file: background_seqs.fasta
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-out',default="background_seqs", help="output file name")
	args = parser.parse_args()
	
	file_name = args.out

	db_to_fasta(file_name)
  
if __name__== "__main__":
	main()


