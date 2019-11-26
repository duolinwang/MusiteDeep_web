import sys
import os.path
import argparse
import re
import itertools
import functions
import numpy as np

def add_inserts(insert_len):
    output = ''
    for i in range(insert_len):
        output += '-'
    
    return output

def add_pad(pad):
    output = ''
    for i in range(pad):
        output += ' '
    return output

def map_seq(starts,lens,seq):
    output = add_pad(starts[0])
    counter = 0
    for i in range(0,len(starts),2):
        
        q_s = starts[i]
        r_s = starts[i+1]
        if q_s == -1:
            output += seq[prev_r_s:r_s]
            prev_r_s = r_s
        elif r_s == -1:
            output += seq[prev_r_s:starts[i+3]]
            i += 2
            prev_r_s = starts[i+1]
            output += add_inserts(lens[2*counter-1])
        else:
            counter += 1
            prev_q_s = starts[i]
            prev_r_s = starts[i+1]
    output += seq[prev_r_s:prev_r_s+lens[-1]]
    return output

#formating string for display
def prepare(id,relative_positions):
	out_data = id
	for i in set(relative_positions): #only keep unique positions
		out_data += ' '+str(i)
	return out_data + '\n'

def display_ptm(ptm,ptm_fp,output):
    """
    write print ready ptm positions to file
    
    parameters:
    ptm: a list contains ptm for each id
    ptm_fp: ptm position file pointer
    
    """
    for id in output:
        out = prepare(id,ptm[id])
        #if len(ptm[id]) > 0:
        #	print(ptm_fp.name+": "+out)
        ptm_fp.write(out)

#def display_output(q_seq,output,identities,fp):
#    """
#    write final display seq output to file
#    
#    parameters:
#    
#    q_seq: user query sequence
#    output: a list of print ready sequences for each id
#    identities: a list of identity for each id
#    fp: output file pointer
#    
#    """
#    q_id = '{:14}'.format("Query_1")
#    fp.write(q_id + q_seq + "\n")
#    for id in output:
#        fp.write('{:14}'.format(id) + output[id] + '{:8}'.format(identities[id]) +  "\n")


def display_output(q_seq,output,identities,fp):
    """
    write final display seq output to file
    
    parameters:
    
    q_seq: user query sequence
    output: a list of print ready sequences for each id
    identities: a list of identity for each id
    fp: output file pointer
    
    """
    q_id = "Query_1"
    lenseq = len(q_seq)
    fp.write(q_id +" identity "+ q_seq + "\n")
    #print(q_id +" identity "+ q_seq + "\n")
    
    for id,identy in sorted(identities.items(),key = lambda item:item[1],reverse=True):
        resseqs = re.sub(' ','-',output[id])
        for x in np.arange(lenseq-len(resseqs)):
            resseqs+='-'
        
        fp.write(id +" "+str(identy)+" "+resseqs+ "\n")
        #print(id +" "+str(identy)+" "+resseqs + "\n")


def get_ptms(starts,lens,ptm,table,seqs):
    ab_ptms = dict()
    for id in seqs:
        ab_ptms[id] = []
        if re.match('(.+)_[1-9]',id):
             real_id = re.match('(.+)_[1-9]',id).group(1)
        else:
            real_id = id
        data = table.find_one({'_id': real_id})
        if ptm in data:
            for i in data[ptm]:
                if int(i) > starts[id][1] and int(i) < (starts[id][-1] + lens[id][-1]):
                    temp_ptm = int(i) - starts[id][1]
                    temp_lens = 0
                    for j in range(0,len(lens[id]),2):
                        temp_lens += lens[id][j]
                        if temp_ptm > temp_lens:
                            #print("id: "+id)
                            #print(i)
                            #print('j: '+str(j))
                            #print(lens[id])
                            #print(starts[id])
                            #print(temp_lens)
                            #print(temp_ptm)
                            if starts[id][j*2+2] == -1: # it is deletion
                                if temp_ptm <= (temp_lens + lens[id][j+1]):
                                    break # ptm is not in seqence
                                else:
                                    temp_ptm -= lens[id][j+1]
                            else: # insertion
                                temp_ptm += lens[id][j+1]
                                temp_lens += lens[id][j+1]
                        else:
                            ab_ptms[id].append(temp_ptm + starts[id][0])
    return ab_ptms

def get_seq(fp):
    output = ''
    line = fp.readline()
    collapsed = ' '.join(line.split())
    data = collapsed.split(" ")
    while data[0] != '}':
        output += re.sub('\"','',data[0])
        line = fp.readline()
        collapsed = ' '.join(line.split())
        data = collapsed.split(" ")
    return output

def blast_output(filepath,ptms,out_folder):
    """
    main function to generate display from blast output
    and write to files
    """
    files = []
    for ptm in ptms:
        files.append(open(out_folder+'/'+ptm+'.txt','w'))
    out_file = open(out_folder+'/blast_output.txt','w')
    
    
    fp = open(filepath)
    table = functions.connectMongoDB('uniprot','table')
    line = fp.readline()
    collapsed = ' '.join(line.split())
    data = collapsed.split(" ")
    door = 0
    output = dict()
    ac = ''
    seq = ''
    counter = dict()
    identities = dict()
    lens = dict()
    starts = dict()
    q_seq = ''
    while line:
        if data[0] == 'seq-data':
            q_seq = re.sub('\"','',data[2]) ###########constructing
            q_seq += get_seq(fp)
        elif data[0] == 'accession':
            ac = re.match(r'\"([A-Z0-9_]+)\",',data[1]).group(1)
            if ac in output:
                seq = table.find_one({'_id': ac})['sequence']
                counter[ac] += 1
                ac = ac + '_' + str(counter[ac])
                output[ac] = ''
            else:
                seq = table.find_one({'_id': ac})['sequence']
                counter[ac] = 1
                output[ac] = ''
            identities[ac] = identity
        #get identities
        if len(data) == 3 and data[2] == '\"num_ident\",':
            line = fp.readline()
            collapsed = ' '.join(line.split())
            data = collapsed.split(" ")
            identity = float(data[2])
        elif len(data) == 3 and data[2] == '\"num_positives\",':
            line = fp.readline()
            collapsed = ' '.join(line.split())
            data = collapsed.split(" ")
            #identity = round(identity/float(data[2]),4)
            identity = int(identity/float(data[2])*100)
        # get starts and lens
        if data[0] == 'starts':
            starts[ac] = []
            door = 1 
        elif data[0] == 'lens':
            door = 2
            lens[ac] = []
        elif data[0] == 'strands': # prepare output
            door = 0
            output[ac] = map_seq(starts[ac],lens[ac],seq)
            #print(ac)
            #print(identities)
            #print(output[ac])
        if door == 1:
            start = re.match(r'(^[-+]?[0-9]+),*$',data[0])
            if start:
                starts[ac].append(int(start.group(1)))
        elif door == 2:
            temp = re.match(r'(^[-+]?[0-9]+),*$',data[0])
            if temp:
                lens[ac].append(int(temp.group(1)))
        line = fp.readline()
        collapsed = ' '.join(line.split())
        data = collapsed.split(" ")
    
    for counter, ptm in enumerate(ptms):
        # generate the ptm position for display
        ab_ptms = get_ptms(starts,lens,ptm,table,output) #TODO finish this function
        display_ptm(ab_ptms,files[counter],output) 
    
    display_output(q_seq,output,identities,out_file)
    out_file.close()
    fp.close()
    for index, ptm in enumerate(ptms):
        files[index].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', default='format11.asn',help="local filepath")
    parser.add_argument('-ptms', default='Phosphotyrosine', help="ptms ptm1_ptm2_ptm3...")
    parser.add_argument('-o', default='display',help="output folder name")
    args = parser.parse_args()
    filepath = args.l
    ptms = args.ptms.split('_')
    out_folder = args.o
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if os.path.exists(filepath):
        blast_output(filepath,ptms,out_folder)	
    else:
        print("File does not exist\n")
        sys.exit()
    
if __name__== "__main__":
    main()