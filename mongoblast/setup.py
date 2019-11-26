# OS: Ubuntu, 18.04.1 LTS
# Python: Python 2.7.15
# Mongodb: v3.2.21 
# Siteng Cai

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-update', type=int, default=0, help="update options: check every # months, default to manual(0)")
    parser.add_argument('-download', type=int, default=0, help="whether to download uniprotData/uniprot.txt from Uniprot/SwissProt (default:0), set to 1 for the first time.")
    args = parser.parse_args()
    step_1 = 'codes/tableGenerator.py'
    step_2 = 'codes/DBtoF.py'
    step_3 = 'codes/annotations.py'
    step_4 = 'makeblastdb -in background_seqs.fasta -dbtype prot -out mydb -parse_seqids'
    subprocess.call(['python3',step_1,'-update',str(args.update),'-download',str(args.download)])
    subprocess.call(['python3',step_2])
    subprocess.call(['python3',step_3])
    subprocess.call([step_4],shell=True)
    print("Setup Finished")


if __name__== "__main__":
	main()