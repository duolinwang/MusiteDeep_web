import os
import time
import numpy as np
import pandas as pd

import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from attention import Attention,myFlatten
from keras.layers.merge import concatenate
from LossCheckPoint import LossModelCheckpoint


#def copy_model(model):
#    config=model.get_config()
#    cp_model = Model.from_config(config)
#    return cp_model

def copy_model(input_row,input_col):
         
         input = Input(shape=(input_row,input_col))
         filtersize1=1
         filtersize2=9
         filtersize3=10
         filter1=200
         filter2=150
         filter3=200
         dropout1=0.75
         dropout2=0.75
         dropout4=0.75
         dropout5=0.75
         dropout6=0
         L1CNN=0
         nb_classes=2
         batch_size=1200
         actfun="relu"; 
         optimization='adam';
         attentionhidden_x=10
         attentionhidden_xr=8
         attention_reg_x=0.151948
         attention_reg_xr=2
         dense_size1=149
         dense_size2=8
         dropout_dense1=0.298224
         dropout_dense2=0
         input = Input(shape=(input_row,input_col))
         x = conv.Convolution1D(filter1, filtersize1,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(input) 
         x = Dropout(dropout1)(x)
         x = Activation(actfun)(x)
         x = conv.Convolution1D(filter2,filtersize2,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(x)
         x = Dropout(dropout2)(x)
         x = Activation(actfun)(x)
         x = conv.Convolution1D(filter3,filtersize3,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(x)
         x = Activation(actfun)(x)
         x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)
         x = Dropout(dropout4)(x)
         x_reshape=Dropout(dropout5)(x_reshape)
         decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
         decoded_x=decoder_x(x)
         output_x = myFlatten(x._keras_shape[2])(decoded_x)
         decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
         decoded_xr=decoder_xr(x_reshape)
         output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)
         output=concatenate([output_x,output_xr])
         output=Dropout(dropout6)(output)
         output=Dense(dense_size1,kernel_initializer='he_normal',activation='relu')(output)
         output=Dropout(dropout_dense1)(output)
         output=Dense(dense_size2,activation="relu",kernel_initializer='he_normal')(output)
         output=Dropout(dropout_dense2)(output)
         out=Dense(nb_classes,kernel_initializer='he_normal',activation='softmax')(output)
         cp_model=Model(input,out)
         return cp_model

def MultiCNN(trainX, trainY,valX=None, valY=None,
             nb_classes=2, nb_epoch=500, earlystop=None,
             weights=None, compiletimes=0,compilemodels=None,
             batch_size=1000, 
             class_weight=None,
             transferlayer=1,forkinase=False,
             predict=False,
             outputweights=None,
             monitor_file=None,
             save_best_only=True,
             load_average_weight=False):
    
    print(trainX.shape)
    if len(trainX.shape)>3:
          trainX.shape=(trainX.shape[0],trainX.shape[2],trainX.shape[3])
    
    if(earlystop is not None): 
        early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
        nb_epoch=10000;#set to a very big value since earlystop used
    
    if(valX is not None):
       print(valX.shape)
       if len(valX.shape)>3:
          valX.shape=(valX.shape[0],valX.shape[2],valX.shape[3])
    
    if compiletimes==0:         
         filtersize1=1
         filtersize2=9
         filtersize3=10
         filter1=200
         filter2=150
         filter3=200
         dropout1=0.75
         dropout2=0.75
         dropout4=0.75
         dropout5=0.75
         dropout6=0
         L1CNN=0
         batch_size=1200
         actfun="relu"; 
         optimization='adam';
         attentionhidden_x=10
         attentionhidden_xr=8
         attention_reg_x=0.151948
         attention_reg_xr=2
         dense_size1=149
         dense_size2=8
         dropout_dense1=0.298224
         dropout_dense2=0
         
         input = Input(shape=(trainX.shape[1],trainX.shape[2]))
         x = conv.Convolution1D(filter1, filtersize1,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(input) 
         x = Dropout(dropout1)(x)
         x = Activation(actfun)(x)
         x = conv.Convolution1D(filter2,filtersize2,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(x)
         x = Dropout(dropout2)(x)
         x = Activation(actfun)(x)
         x = conv.Convolution1D(filter3,filtersize3,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(x)
         x = Activation(actfun)(x)
         x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)
         
         x = Dropout(dropout4)(x)
         x_reshape=Dropout(dropout5)(x_reshape)
         
         decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
         decoded_x=decoder_x(x)
         output_x = myFlatten(x._keras_shape[2])(decoded_x)
         
         decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
         decoded_xr=decoder_xr(x_reshape)
         output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)
         
         output=concatenate([output_x,output_xr])
         output=Dropout(dropout6)(output)
         output=Dense(dense_size1,kernel_initializer='he_normal',activation='relu')(output)
         output=Dropout(dropout_dense1)(output)
         output=Dense(dense_size2,activation="relu",kernel_initializer='he_normal')(output)
         output=Dropout(dropout_dense2)(output)
         out=Dense(nb_classes,kernel_initializer='he_normal',activation='softmax')(output)
         cnn=Model(input,out)
         cnn.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
         
    else:
         cnn=compilemodels
    
    if(predict is False):
         if(weights is not None and compiletimes==0): #for the first time
            print("load weights:"+weights)
            if not forkinase:
                 cnn.load_weights(weights);
            else:
                 #cnn2=copy_model(cnn)
                 cnn2=copy_model(trainX.shape[1],trainX.shape[2])
                 cnn2.load_weights(weights);
                 for l in range((len(cnn2.layers)-transferlayer)): #the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())
                    #cnn.layers[l].trainable= False  # for frozen layer
         
         print("##################save_best_only "+str(save_best_only))
         weight_checkpointer = LossModelCheckpoint(
                                model_file_path =outputweights+'_iteration'+str(compiletimes),
                                monitor_file_path= monitor_file +'_iteration' + str(compiletimes)+'.json',
                                verbose=1,save_best_only=save_best_only,
                                monitor='val_loss',mode='min',
                                save_weights_only=True)
         
         if(valX is not None):
             if(earlystop is None):
               fitHistory = cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(valX, valY),callbacks=[weight_checkpointer])
             else:
               fitHistory = cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(valX, valY), callbacks=[early_stopping,weight_checkpointer])
         else:
             fitHistory = cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch,callbacks=[weight_checkpointer])
         
         if load_average_weight:
            if save_best_only:
               last_weights =cnn.get_weights()
               cnn.load_weights(outputweights+'_iteration'+str(compiletimes)) #every iteration need to reload the best model for next run
               saved_weights = cnn.get_weights()
               avg_merged_weights = list()
               for layer in range(len(last_weights)):
                   avg_merged_weights.append(1/2*(last_weights[layer]+saved_weights[layer]))
               
               cnn.set_weights(avg_merged_weights)
         else:
            cnn.load_weights(outputweights+'_iteration'+str(compiletimes)) #every iteration need to reload the best model for next run
    
    return cnn
