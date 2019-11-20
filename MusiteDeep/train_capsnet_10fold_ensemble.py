import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
import pandas as pd
import numpy as np
import argparse
from Bootstrapping_capsnet_callback import bootStrapping_allneg_continue_keras2
from EXtractfragment_sort import extractFragforTraining
import timeit
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.special import softmax
import json
import keras.backend as K

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_data(inputfile,window,residues):
    allfrag=extractFragforTraining(inputfile,window,'-',focus=residues)
    allfrag=allfrag.as_matrix()
    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    folds = kf.split(allfrag)
    return allfrag,folds


def calculate_avg_weights(inputweights,model_arch):
    callback_weights=list()
    all_weights = list()
    n_models = 0
    filename = inputweights +  str(n_models)
    callbackfile = filename+".json"
    while os.path.exists(filename):
      # define filename for this ensemble
      model_arch.load_weights(filename)
      # add to list of members
      all_weights.append(model_arch.get_weights())
      print('>loaded %s' % filename)
      with open(callbackfile) as checkpoint_fp:
           callback_weights.append(1/float(json.load(checkpoint_fp)["val_loss"])+0.00001)
      
      n_models+=1
      print("deleting "+filename)
      os.remove(filename)
      os.remove(callbackfile)
      
      filename = inputweights +  str(n_models)
      callbackfile=filename+".json"
    
    callback_weights = softmax(callback_weights)
    n_layers = len(all_weights[0])
    avg_model_weights = list()
    for layer in range(n_layers):
      layer_weights = np.array([x[layer] for x in all_weights])
      avg_layer_weights = np.average(layer_weights, axis=0, weights=callback_weights)
      avg_model_weights.append(avg_layer_weights)
    
    return avg_model_weights


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-input',  dest='inputfile', type=str, help='training data in fasta format. Sites followed by "#" are positive sites for a specific PTM prediction.', required=True)
    parser.add_argument('-output',  dest='outputfolder', type=str, help='folder for the output models (model and parameter files).', required=True)
    parser.add_argument('-residue-types',  dest='residues', type=str, help='Residue types that this model focus on. For multiple residues, seperate each with \',\'. \n\
    Note: all the residues specified by this parameter will be trained in one model.', required=True)
    parser.add_argument('-nclass',  dest='nclass', type=int, help='number of classifiers to be trained for one time. [Default:1]', required=False, default=1)
    parser.add_argument('-window',  dest='window', type=int, help='window size: the number of amino acid of the left part or right part adjacent to a potential PTM site. 2*\'windo size\'+1 amino acid will be extracted for one protential fragment. [Default:16]', required=False, default=16)
    parser.add_argument('-maxneg',  dest='maxneg', type=int, help='maximum iterations for each classifier which controls the maximum copy number of the negative data which has the same size with the positive data. [Default: 30]', required=False, default=30)
    parser.add_argument('-nb_epoch',  dest='nb_epoch', type=int, help='number of epoches for one bootstrap step. It is invalidate, if earlystop is set.', required=False, default=None)
    parser.add_argument('-earlystop',  dest='earlystop', type=int, help='after the \'earlystop\' number of epochs with no improvement the training will be stopped for one bootstrap step. [Default: 20]', required=False, default=20)
    parser.add_argument('-inputweights',  dest='inputweights', type=str, help='Initial weights saved in a HDF5 file.', required=False, default=None)
    parser.add_argument('-checkpointweights',  dest='checkpointweights', type=str, help='folder for the intermediate checkpoint files.', required=True, default=None)
    parser.add_argument('-save_best_only', dest='save_best_only',action='store_true',help="save the best models in each iteration if -load_average_weight is set, this must be set.")
    parser.add_argument('-load_average_weight',dest='load_average_weight', action="store_true",help="whether load the average weights of best and last model, if seted the save_best_only will be changed to True")
    parser.add_argument('-balance_val',dest='balance_val', action="store_true",help="use the balanced validation set to monitor the training process")
    args = parser.parse_args()
    
    if args.load_average_weight: #if load_average_weight has been set, save_best_only must be set 
           args.save_best_only=True
           print("Because load_average_weight has been set, changing save_best_only to "+str(args.save_best_only))
    
    residues=args.residues.split(",")
    ensure_dir(args.outputfolder)
    ensure_dir(args.checkpointweights)
    
    outputmodel = args.outputfolder+str("model_HDF5model")
    outputparameter = args.outputfolder+str("model_parameters")
    codemode=0 #coding method
    model='nogradientstop' #use this model
    nb_classes=2 # binary classification
    
        
    for k, v in vars(args).items():
        print(k, ':', v)
    
    
    try:
       output = open(outputparameter, 'w')
    except IOError:
       print('cannot write to ' + outputparameter+ "!\n")
       exit()
    else:
       output.write("%d\t%d\t%s\tgeneral\t%d\t%s\t%d" % (args.nclass,args.window,args.residues,codemode,model,nb_classes))
    
    start = timeit.default_timer()
    allfrag,folds=preprocess_data(args.inputfile,args.window,residues)
    for i,(train_indices,val_indices) in enumerate(folds):
        for bt in range(args.nclass):
            checkpointoutput=args.checkpointweights+"weights_fold"+str(i)+"_nclass"+str(bt)
            models=bootStrapping_allneg_continue_keras2(allfrag[train_indices],allfrag[val_indices],
                                                      srate=1,nb_epoch1=1,nb_epoch2=args.nb_epoch,earlystop=args.earlystop,maxneg=args.maxneg,
                                                      outputweights=checkpointoutput,
                                                      monitor_file_name = checkpointoutput,
                                                      inputweights=args.inputweights,
                                                      balance_validation=args.balance_val,
                                                      model=model,
                                                      codingMode=codemode,
                                                      nb_classes=nb_classes,
                                                      save_best_only=args.save_best_only,
                                                      load_average_weight=args.load_average_weight
                                                      )
            
            avg_model_weights = calculate_avg_weights(checkpointoutput+"_iteration",models)
            models.set_weights(avg_model_weights)
            models.save_weights(outputmodel+'_fold'+str(i)+'_class'+str(bt),overwrite=True)#only keep this averaged model and delete all the other models
            K.clear_session()
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)
   