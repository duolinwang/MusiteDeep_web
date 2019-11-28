import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf

import pandas as pd
import numpy as np
import argparse
import csv
from DProcess import convertRawToXY
from capsulenet_callback import Capsnet_main
from multiCNN_callback import MultiCNN
from EXtractfragment_sort import extractFragforMultipredict

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory !='':
       if not os.path.exists(directory):
          os.makedirs(directory)

def batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def evaluate(predict_proba,testY):
    from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef
    true_label = [np.argmax(x) for x in testY]
    roc_score=roc_auc_score(true_label,predict_proba)
    pr_score=average_precision_score(true_label,predict_proba)
    pred_label = [np.argmax([0.5,x]) for x in predict_proba]
    mcc = matthews_corrcoef(true_label,pred_label)
    return roc_score,pr_score,mcc


def batch_predict(data,arch_cnn,arch_caps_caps,model_cnn,model_caps,nclass,outputfile,foldnum,num_ptms,ptmtype,nclass_init=None):
    predictproba=np.zeros((len(data),1))
    batch_size=500
    totalindex = int(np.ceil(float(len(data)/batch_size)))
    totalnum=totalindex*foldnum
    processed_num=0
    batch_generator = batch(data,batch_size)
    y_label=[]
    for index in range(totalindex):
      websiteoutput = open(outputfile+"_predicted_num.txt",'w')
      prossratio = round(float(index)/(totalindex)*100,2);
      print(ptmtype+":"+str(prossratio)+"%\n")
      websiteoutput.write(ptmtype+":"+str(prossratio)+"\n")
      websiteoutput.close()
      
      batch_data = next(batch_generator)
      testdata,tempy=convertRawToXY(batch_data.values,codingMode=0)
      y_label.extend(tempy)
      testdata.shape=(testdata.shape[0],testdata.shape[2],testdata.shape[3])
      
      for ini in range(foldnum):
        for bt in range(nclass):     
            arch_cnn.load_weights(model_cnn+'_fold'+str(ini)+'_class'+str(bt))
            predictproba[index*batch_size:index*batch_size+len(batch_data)] += arch_cnn.predict(testdata,batch_size=batch_size)[:,1].reshape(-1,1)
            arch_caps.load_weights(model_caps+'_fold'+str(ini)+'_class'+str(bt))
            predictproba[index*batch_size:index*batch_size+len(batch_data)] += arch_caps.predict(testdata,batch_size=batch_size)[:,1].reshape(-1,1)
    
    
    return predictproba/(2*nclass*foldnum),y_label

class ProtIDResult(object):
    """docstring for ClassName"""
    def __init__(self, prot_id, scores=list(), residues=list(), positions=list(), ptmtypes=list()):
        super(ProtIDResult, self).__init__()
        self.prot_id = prot_id
        self.residues_dic = {prot_id:{}} #only one per postion!
        self.scores_dic = {prot_id:{}} #all the following three use prot_id and position as key 
        self.ptmtypes_dic = {prot_id:{}}
    
    def update(self, prot_id, scores, residues, positions, ptmtypes):
        if positions not in self.ptmtypes_dic[prot_id]:
            self.ptmtypes_dic[prot_id][positions] = [ptmtypes] #arrary to keep the scores,residues and ptmtypes in the same order.
        elif ptmtypes in self.ptmtypes_dic[prot_id][positions]: # will never occur! no duplicate ptm types for same prot_id and same position
            print("duplicate resulte!\n")
            return
        else:self.ptmtypes_dic[prot_id][positions] += [ptmtypes]
        
        if positions not in self.scores_dic[prot_id]:
            self.scores_dic[prot_id][positions] = [round(scores,3)] 
        else:self.scores_dic[prot_id][positions] += [round(scores,3)] #add new score for this position
        
        if positions not in self.residues_dic[prot_id]: # 
            self.residues_dic[prot_id][positions]  = residues #only one residue shown for one position!
    
    def __str__(self):
        res_str = ""
        defaultcutoff=0.5;
        res_str = res_str+self.prot_id+"\n"
        for pos in sorted(self.scores_dic[self.prot_id].keys()): #order by pos, it is the key for scores_dic[prot_id]
            #res_str = res_str+"\""+self.prot_id+"\"\t"
            res_str = res_str+str(pos)+"\t"
            #res_str = res_str+str(','.join(self.residues_dic[self.prot_id][pos]))+"\t"
            res_str = res_str+str(self.residues_dic[self.prot_id][pos])+"\t" # no need to add \"
            #res_str = res_str+str(','.join([str(x) for x in self.scores_dic[self.prot_id][pos]]))+"\t"
            #res_str = res_str+str(','.join(self.ptmtypes_dic[self.prot_id][pos]))+"\t"
            ptms=[]
            pastptms=[]
            for index in range(len(self.scores_dic[self.prot_id][pos])):
                if self.ptmtypes_dic[self.prot_id][pos][index] =='Phosphoserine_Phosphothreonine':
                    if self.residues_dic[self.prot_id][pos] == 'S':
                       ptmln="Phosphoserine:"+str(self.scores_dic[self.prot_id][pos][index])
                    else:
                       ptmln="Phosphothreonine:"+str(self.scores_dic[self.prot_id][pos][index])
                else:
                    ptmln=self.ptmtypes_dic[self.prot_id][pos][index]+":"+str(self.scores_dic[self.prot_id][pos][index])
                
                ptms.append(ptmln)
                if self.scores_dic[self.prot_id][pos][index] > defaultcutoff:
                       pastptms.append(ptmln)
            
            res_str+=';'.join(ptms)+"\t"
            if len(pastptms)>0:
                 res_str+=';'.join(pastptms)
            else:
                 res_str+="None"
            
            res_str+="\n"
        return res_str

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='MusiteDeep prediction tool for general, kinase-specific phosphorylation prediction or custom PTM prediction by using custom models.')
    parser.add_argument('-input',  dest='inputfile', type=str, help='Protein sequences to be predicted in FASTA format.', required=True)
    parser.add_argument('-output',  dest='output', type=str, help='prefix of the prediction results.', required=True)
    parser.add_argument('-model-prefix',  dest='modelprefix', type=str, help='prefix of custom model used for prediciton. If donnot have one, please run train_capsnet_10fold_ensemble.py and train_CNN_10fold_ensemble to train models for a particular PTM type.', required=False,default=None)
    args = parser.parse_args()
    
    ensure_dir(args.output) #mkfolders for outputfile
    websiteoutput = open(args.output+"_predicted_num.txt",'w') #generate this file at the beginning! 
    websiteoutput.write("Start:0\n")
    websiteoutput.close()
    if args.modelprefix is None:
       print("If you want to do prediction by a custom model, please specify the prefix for an existing custom model by -model-prefix!\n\
       It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.\n \
       If you don't have such files, please run train_general.py or train_kinase.py to get the custom model first!\n"
       )
       exit()
    else: #custom prediction
      foldername,filename = os.path.split(args.modelprefix)
      print(foldername)
      print(filename)
      modeloptins =filename.split(";")
      print("modeloptions="+str(modeloptins))
      num_ptms = len(modeloptins)
      prot_id_dic={}
      ptmindex=0
      #if the window size is not fixed as 16, move this below window=int(parameters.split("\t")[1])
      arch_caps = Capsnet_main(np.zeros([3,2*16+1,21]),[],nb_epoch=1,compiletimes=0,lr=0.001,batch_size=500,lam_recon=0,routings=3,modeltype='nogradientstop',nb_classes=2,predict=True)
      arch_cnn = MultiCNN(np.zeros([3,2*16+1,21]),[],nb_epoch=1,compiletimes=0,batch_size=500,nb_classes=2,predict=True)
      
      for ptmtype in modeloptins:
        model_cnn=os.path.join(foldername,ptmtype,"CNNmodels","model"+str("_HDF5model"))
        model_caps=os.path.join(foldername,ptmtype,"capsmodels","model"+str("_HDF5model"))
        parameter=os.path.join(foldername,ptmtype,"capsmodels","model"+str("_parameters"))
        try:
            f=open(parameter,'r')
        except IOError:
            print('cannot open '+ parameter+" ! check if the model exists. please run train_general.py or train_kinase.py to get the custom model first!\n")
        else:
             f= open(parameter, 'r')
             parameters=f.read()
             f.close()
        
        nclass=int(parameters.split("\t")[0])
        window=int(parameters.split("\t")[1])
        residues=parameters.split("\t")[2]
        residues=residues.split(",")
        testfrag,ids,poses,focuses,idlist=extractFragforMultipredict(args.inputfile,window,'-',focus=residues)
        foldnum=10
        predictproba,y_true=batch_predict(testfrag,arch_cnn,arch_caps,model_cnn,model_caps,nclass,args.output,foldnum,num_ptms,ptmtype)           
        poses=poses+1;
        ptmindex+=1
        #results=np.column_stack((ids,poses,focuses,predictproba))
        #result=pd.DataFrame(results)
        #result.to_csv(args.output+"_"+ptmtype+".txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
        print("Successfully predicted from model:"+ptmtype+"!\n")
        for i in range(len(ids)):
            prot_id=ids.values[i][0]
            #print("prot_id is "+prot_id)
            if prot_id not in prot_id_dic:
                    prot_id_dic[prot_id] = ProtIDResult(prot_id)
            
            prot_id_dic[prot_id].update(prot_id, predictproba[i][0], focuses.values[i][0], poses.values[i][0], ptmtype)
        
      
      #write results to file
      target = open(args.output+"_results.txt", "w")
      #add a header
      target.write("Position\tResidue\tPTMscores\tCutoff=0.5\n")
      for prot_id in idlist: #must keep the order!
          if prot_id in prot_id_dic.keys(): #only print proteins in the result files. some protein do results dont print them!
                v = str(prot_id_dic[prot_id])
                target.write(v)
      
      target.close()
      websiteoutput = open(args.output+"_predicted_num.txt",'w')
      websiteoutput.write("All:100\n")
      websiteoutput.close()
      

#test in musiteDeepCapsnet
#python3 predict_multi_batch.py -input test_SUMO.fasta -output test_SUMO_multipredict -model-prefix "./models//Ubiquitination-K|SUMOylation-K|N6-acetyllysine|Methyllysine-K"
