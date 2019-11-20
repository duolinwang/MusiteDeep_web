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
from EXtractfragment_sort import extractFragforPredict

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


def batch_predict(data,models,model,nclass,outputfile,nclass_init=None):
    predictproba=np.zeros((len(data),1))
    batch_size=500
    totalindex = int(np.ceil(float(len(data)/batch_size)))
    batch_generator = batch(data,batch_size)
    y_label=[]
    for index in range(totalindex):
      if (index+1) % 2 ==0:
        websiteoutput = open(outputfile+"_predicted_num.txt",'w')
        prossratio = float(index)/totalindex*100
        websiteoutput.write(str(prossratio)+"\n")
        websiteoutput.close()
      
      batch_data = next(batch_generator)
      testdata,tempy=convertRawToXY(batch_data.values,codingMode=0)
      y_label.extend(tempy)
      testdata.shape=(testdata.shape[0],testdata.shape[2],testdata.shape[3])
      
      for ini in range(foldnum):
                for bt in range(nclass):     
                    models.load_weights(model+'_fold'+str(ini)+'_class'+str(bt))
                    predictproba[index*batch_size:index*batch_size+len(batch_data)] += models.predict(testdata,batch_size=batch_size)[:,1].reshape(-1,1)
    
    return predictproba/(nclass*foldnum),y_label


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='MusiteDeep prediction tool for general, kinase-specific phosphorylation prediction or custom PTM prediction by using custom models.')
    parser.add_argument('-input',  dest='inputfile', type=str, help='Protein sequences to be predicted in fasta format.', required=True)
    parser.add_argument('-output',  dest='output', type=str, help='prefix of the prediction results.', required=True)
    parser.add_argument('-model-prefix',  dest='modelprefix', type=str, help='prefix of custom model used for prediciton. If donnot have one, please run train_general.py to train a custom general PTM model or run train_kinase.py to train a custom kinase-specific PTM model.', required=False,default=None)
    parser.add_argument('-residue-types',  dest='residues', type=str, help='Residue types that to be predicted, only used when -predict-type is \'general\'. For multiple residues, seperate each with \',\'',required=False,default="S,T,Y")
    parser.add_argument('-evaluation',  dest='evaluation', action="store_true", 
                        help='Use some evaluation metrics to evaluate the results. If seted, the input file must contains # as positive annotations, please refer to the format of annotation in MusiteDeep.')
    
    args = parser.parse_args()
    
    ensure_dir(args.output) #mkfolders for outputfile
    residues=args.residues.split(",")
    
    if args.modelprefix is None:
       print("If you want to do prediction by a custom model, please specify the prefix for an existing custom model by -model-prefix!\n\
       It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.\n \
       If you don't have such files, please run train_general.py or train_kinase.py to get the custom model first!\n"
       )
       exit()
    else: #custom prediction
      model=args.modelprefix+str("_HDF5model")
      parameter=args.modelprefix+str("_parameters")
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
      testfrag,ids,poses,focuses=extractFragforPredict(args.inputfile,window,'-',focus=residues)
      models = Capsnet_main(np.zeros([3,2*16+1,21]),[],nb_epoch=1,compiletimes=0,lr=0.001,batch_size=500,lam_recon=0,routings=3,modeltype='nogradientstop',nb_classes=2,predict=True)
      foldnum=10
      predictproba,y_true=batch_predict(testfrag,models,model,nclass,args.output,foldnum)           
      poses=poses+1;
      results=np.column_stack((ids,poses,focuses,predictproba))
      result=pd.DataFrame(results)
      result.to_csv(args.output+"_results.txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
      print("Successfully predicted from custom models !\n")
      websiteoutput = open(args.output+"_predicted_num.txt",'w')
      websiteoutput.write("100\n")
      websiteoutput.close()
      if args.evaluation:
          #try: 
          roc_score,pr_score,mcc=evaluate(predictproba,y_true)
          print("AUC score:%0.3f" %(roc_score))
          print("Average_precision_score:%0.3f" %(pr_score))
          print("MCC:%0.3f" %(mcc))
          with open(args.output+"_evaluation.txt",'w') as out:
               out.write("AUC score:%0.3f\n" %(roc_score))
               out.write("Average_precision_score:%0.3f\n" %(pr_score))
               out.write("MCC:%0.3f\n" %(mcc))
          
          #except:
          #      if(np.sum(y_true)==0):
          #           print("There is no ground truth annotations in the input data, so cannot do evaluation.")
          #      else:
          #          print("Error during evaluation.")
   