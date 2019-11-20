from DProcess import convertRawToXY
import pandas as pd
import numpy as np
import keras.models as models
import sys
from multiCNN_callback import MultiCNN


def bootStrapping_allneg_continue_keras2(trainfile,valfile=None,srate=0.8,
                                         nb_epoch1=3,nb_epoch2=30,earlystop=None,
                                         maxneg=None,codingMode=0,
                                         inputweights=None,outputweights=None,nb_classes=2,
                                         transferlayer=0,forkinase=False,
                                         predict=False,
                                         balance_validation=True,
                                         monitor_file_name=None,
                                         save_best_only=True,
                                         load_average_weight=False):
  trainX = trainfile
  for i in range(len(trainX)):
      trainX[i,0]=int(trainX[i,0])
  
  train_pos=trainX[np.where(trainX[:,0]!=0)]
  train_neg=trainX[np.where(trainX[:,0]==0)]
  train_pos=pd.DataFrame(train_pos)
  train_neg=pd.DataFrame(train_neg)
  train_pos_s=train_pos.sample(train_pos.shape[0]); #shuffle train pos
  train_neg_s=train_neg.sample(train_neg.shape[0]); #shuffle train neg
  slength=int(train_pos.shape[0]*srate);
  nclass=int(train_neg.shape[0]/slength);
  if(valfile is not None): # use all data in valfile as val
     valX = valfile
     for i in range(len(valX)):
         valX[i,0]=int(valX[i,0])
     
     val_pos=valX[np.where(valX[:,0]!=0)]
     val_neg=valX[np.where(valX[:,0]==0)]
     val_pos=pd.DataFrame(val_pos)
     val_neg=pd.DataFrame(val_neg)
     if balance_validation:#extract balanced val_neg
        val_neg=val_neg.sample(val_pos.shape[0]) 
     
     val_all=pd.concat([val_pos,val_neg])
     val_all_s = val_all.sample(val_all.shape[0])
     valX1,valY1 = convertRawToXY(val_all_s.as_matrix(),codingMode=codingMode)
  else:     #selct 0.1 samples of training data as val
            a=int(train_pos.shape[0]*0.9);
            b=train_neg.shape[0]-int(train_pos.shape[0]*0.1);
            print ("train pos="+str(train_pos.shape[0])+str('\n'))
            print ("train neg="+str(train_neg.shape[0])+str('\n'))
            print (" a="+str(a)+" b="+str(b)+str('\n'))
            train_pos_s=train_pos[0:a]
            train_neg_s=train_neg[0:b]
            print ("train pos s="+str(train_pos_s.shape[0])+str('\n'))
            print ("train neg s="+str(train_neg_s.shape[0])+str('\n'))
            
            val_pos=train_pos[(a+1):];
            print ("val_pos="+str(val_pos.shape[0])+str('\n'))
            val_neg=train_neg[b+1:];
            print ("val_neg="+str(val_neg.shape[0])+str('\n'))
            
            val_all=pd.concat([val_pos,val_neg])
            val_all_s = val_all.sample(val_all.shape[0])
            valX1,valY1 = convertRawToXY(val_all_s.as_matrix(),codingMode=codingMode)
            slength=int(train_pos_s.shape[0]*srate); #update slength
            nclass=int(train_neg_s.shape[0]/slength);
            
  if(maxneg is not None):
       nclass=min(maxneg, nclass); #cannot do more than maxneg times
  
  #modelweights=None;
  for I in range(nb_epoch1):
    train_neg_s=train_neg_s.sample(train_neg_s.shape[0]) #shuffle neg sample
    train_pos_ss=train_pos_s.sample(slength)
    for t in range(nclass):
        train_neg_ss=train_neg_s[(slength*t):(slength*t+slength)];
        train_all=pd.concat([train_pos_ss,train_neg_ss])
        train_all_shuffle = train_all.sample(train_all.shape[0])
        trainX1,trainY1 = convertRawToXY(train_all_shuffle.as_matrix(),codingMode=codingMode)
        if t==0:
            models=MultiCNN(trainX=trainX1,trainY=trainY1,valX=valX1,valY=valY1,nb_classes=nb_classes,nb_epoch=nb_epoch2,earlystop=earlystop,weights=inputweights,compiletimes=t,batch_size=1000,class_weight=None,transferlayer=transferlayer,forkinase=forkinase,predict=predict,outputweights=outputweights,monitor_file=monitor_file_name,save_best_only=save_best_only,load_average_weight=load_average_weight)
        else:
            models=MultiCNN(trainX=trainX1,trainY=trainY1,valX=valX1,valY=valY1,nb_classes=nb_classes,nb_epoch=nb_epoch2,earlystop=earlystop,weights=inputweights,compiletimes=t,compilemodels=models,batch_size=1000,class_weight=None,transferlayer=transferlayer,forkinase=forkinase,outputweights=outputweights,monitor_file=monitor_file_name,save_best_only=save_best_only,load_average_weight=load_average_weight)
        
        #save weights already does in callbacks in MultiCNN
        print ("modelweights assigned for "+str(I)+" and "+str(t)+"\n")
        #if(outputweights is not None):
        #    models.save_weights(outputweights+ '_iteration'+str(t),
        #                        overwrite=True) #already in callbacks
  
  
  return models