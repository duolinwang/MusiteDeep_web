# mongoBlast
OS: Ubuntu 16.04.5 LTS
Python: Python 3.5 
Mongodb: v3.2.22  

Install MongoDB:  
sudo apt install mongodb-server-core  
or
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv D68FA50FEA312927
echo "deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo mkdir -p /data/db  
sudo chmod 777 /data/db

Install Blast:
wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/ncbi-blast-2.7.1+-x64-linux.tar.gz  
tar zxvpf ncbi-blast-2.7.1+-x64-linux.tar.gz  
export PATH=$PATH:$HOME/ncbi-blast-2.7.1+/bin:$PATH  

Install Crontab for auto update the database. Can be skipped if the auto update is not enabled.:  
pip3 uninstall crontab  
pip3 install python-crontab  

Install required modules:
pip install feedparser
pip install pymongo
pip install configparser

Run:  
1.mongod  --port 23333
2.For first time setup, run (takes about 30 minutes): 

python setup.py -update 1 -download 1

3.For one query sequence: 
python user_query.py -query [input sequence in FASTA format] -ptms [PTMs to be annotated.]  -o [output folder]
For multiple ones, use \"_\" to seperate them. For example: -ptms ptm1_ptm2_ptm3

4.Outputs:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  blastresult.txt
The first two lines represent the ID and sequence of the query sequence, 
and the following lines are the aligned sequences by Blast with the PTM annotations 
in the Uniprot/Swiss-Prot database. Each alignment result consists of two lines: 
the first line contains the Uniprot sequence access ID, 
the Blast identity in the parenthesis, 
the selected PTM types for annotations and the corresponding positions 
(according to the position of the query sequence) that have the annotations; 
the second line contains the aligned amino acids, with hyphens "-" representing the unaligned ones.


