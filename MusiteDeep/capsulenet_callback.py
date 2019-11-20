import numpy as np
from keras import layers,optimizers
from keras.models import *
from keras import backend as K
from keras.utils import to_categorical
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.layers import Dropout,Activation,Add
K.set_image_data_format('channels_last')
from capsulelayers import CapsuleLayer,CapsuleLayer_nogradient_stop,PrimaryCap,  Length, Mask
from LossCheckPoint import LossModelCheckpoint
from keras.engine.topology import Layer
from keras.regularizers import l1,l2,l1_l2
from keras.layers.normalization import BatchNormalization

class LearningRate(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.learningrate = 0
    
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        self.learningrate=lr

class Extract_outputs(Layer):
    def __init__(self,outputdim=0, **kwargs):
        #self.input_spec = [InputSpec(ndim='3+')]
        self.outputdim=outputdim
        super(Extract_outputs, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return tuple([None,input_shape[1], self.outputdim])
    
    def call(self, x, mask=None):
        x=x[:,:,:self.outputdim]
        #return K.batch_flatten(x)
        return x
    
    def get_config(self):
        config = {
        'outputdim': self.outputdim
        
        }
        base_config = super(Extract_outputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Extract_weight_c(Layer):
    def __init__(self,outputdim, **kwargs):
        #self.input_spec = [InputSpec(ndim='3+')]
        self.outputdim=outputdim
        super(Extract_weight_c, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return tuple([None,input_shape[1], input_shape[-1]-self.outputdim])
    
    def call(self, x, mask=None):
        x=x[:,:,self.outputdim:]
        #return K.batch_flatten(x)
        return x
    
    def get_config(self):
        config = {
        'outputdim': self.outputdim
        
        }
        base_config = super(Extract_weight_c, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def custom_binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.flatten(y_true)), axis=-1)

def CapsNet(input_shape, n_class, routings,modeltype,power=2):
    if modeltype == "nogradientstop":
       return CapsNet_nogradientstop(input_shape,n_class,routings)
    if modeltype == "nogradientstop_crossentropy":
           return CapsNet_nogradientstop_crossentropy(input_shape,n_class,routings)

def CapsNet_nogradientstop(input_shape, n_class, routings): # best testing results! val 0.13xx testX cnn1 200 1 cnn2 150 9 drop1 0.68 drop20.68 n_channels 50 kernel_size 20,dropout1
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal',activation='relu', name='conv1')(x)
    #conv1=BatchNormalization()(conv1)
    conv1 = Dropout(0.7)(conv1)
    conv2 = layers.Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal',activation='relu', name='conv2')(conv1)
    #conv1=BatchNormalization()(conv1)
    conv2 = Dropout(0.75)(conv2) #0.75 valx loss has 0.1278!
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=60, kernel_size=20, kernel_initializer='he_normal',strides=1, padding='valid',dropout=0.2)
    dim_capsule_dim2=10
    #Capsule layer. Routing algorithm works here.
    digitcaps_c = CapsuleLayer_nogradient_stop(num_capsule=n_class, dim_capsule=dim_capsule_dim2, num_routing=routings,name='digitcaps',kernel_initializer='he_normal',dropout=0.1)(primarycaps)
    #digitcaps_c = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule_dim2, num_routing=routings,name='digitcaps',kernel_initializer='he_normal')(primarycaps)
    digitcaps = Extract_outputs(dim_capsule_dim2)(digitcaps_c)
    weight_c  = Extract_weight_c(dim_capsule_dim2)(digitcaps_c)
    out_caps = Length(name='capsnet')(digitcaps)
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction
    
    # Shared Decoder model in training and prediction
    decoder = Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=dim_capsule_dim2*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    
    # Models for training and evaluation (prediction)
    #train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
    #eval_model = Model(x, [out_caps, decoder(masked)])
    eval_model = Model(x, out_caps)
    #weight_c_model = Model(x,weight_c)
    # manipulate model
    #noise = layers.Input(shape=(n_class, dim_capsule_dim2))
    #noised_digitcaps = layers.Add()([digitcaps, noise])
    #masked_noised_y = Mask()([noised_digitcaps, y])
    #manipulate_model = Model([x, y, noise], decoder(masked_noised_y))
    #return train_model, eval_model, manipulate_model,weight_c_model
    return eval_model

def CapsNet_nogradientstop_crossentropy(input_shape, n_class, routings): # best testing results! val 0.13xx testX cnn1 200 1 cnn2 150 9 drop1 0.68 drop20.68 n_channels 50 kernel_size 20,dropout1
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv1D(filters=200, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal',activation='relu', name='conv1')(x)
    #conv1=BatchNormalization()(conv1)
    conv1 = Dropout(0.7)(conv1)
    conv2 = layers.Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal',activation='relu', name='conv2')(conv1)
    #conv1=BatchNormalization()(conv1)
    conv2 = Dropout(0.75)(conv2) #0.75 valx loss has 0.1278!
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=60, kernel_size=20, kernel_initializer='he_normal',strides=1, padding='valid',dropout=0.2)
    dim_capsule_dim2=10
    # Capsule layer. Routing algorithm works here.
    digitcaps_c = CapsuleLayer_nogradient_stop(num_capsule=n_class, dim_capsule=dim_capsule_dim2, num_routing=routings,name='digitcaps',kernel_initializer='he_normal',dropout=0.1)(primarycaps)
    #digitcaps_c = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule_dim2, num_routing=routings,name='digitcaps',kernel_initializer='he_normal')(primarycaps)
    digitcaps = Extract_outputs(dim_capsule_dim2)(digitcaps_c)
    weight_c  = Extract_weight_c(dim_capsule_dim2)(digitcaps_c)
    out_caps = Length()(digitcaps)
    out_caps = Activation('softmax',name='capsnet')(out_caps)
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction
    
    # Shared Decoder model in training and prediction
    decoder = Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=dim_capsule_dim2*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    
    # Models for training and evaluation (prediction)
    #train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
    #eval_model = Model(x, [out_caps, decoder(masked)])
    eval_model = Model(x, out_caps)
    #weight_c_model = Model(x,weight_c)
    # manipulate model
    #noise = layers.Input(shape=(n_class, dim_capsule_dim2))
    #noised_digitcaps = layers.Add()([digitcaps, noise])
    #masked_noised_y = Mask()([noised_digitcaps, y])
    #manipulate_model = Model([x, y, noise], decoder(masked_noised_y))
    #return train_model, eval_model, manipulate_model,weight_c_model
    return eval_model
    

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    
    return K.mean(K.sum(L, 1))

def speard_loss(m):
    def loss(y_true,y_pred):
          L = K.square(K.maximum(0.,m-(y_true-y_pred)))
          L=K.mean(K.sum(L, 1))-m**2
          return L
    return loss

def Capsnet_main(trainX,trainY,valX=None,valY=None,nb_classes=2,nb_epoch=500,earlystop=None,weights=None,compiletimes=0,compilemodels=None,lr=0.001,lrdecay=1,batch_size=500,lam_recon=0.392,routings=3,modeltype=5,class_weight=None,activefun='linear',power=2,predict=False,outputweights=None,monitor_file=None,save_best_only=True,load_average_weight=False):
    print(trainX.shape)
    if len(trainX.shape)>3:
          trainX.shape=(trainX.shape[0],trainX.shape[2],trainX.shape[3])
    
    if(valX is not None):
       print(valX.shape)
       if len(valX.shape)>3:
          valX.shape=(valX.shape[0],valX.shape[2],valX.shape[3])
    
    if(earlystop is not None):#use early_stop to control nb_epoch there must contain a validation if not provided will select one
          early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
          nb_epoch=10000
    
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (lrdecay ** epoch))
    if compiletimes==0:
        model = CapsNet(input_shape=trainX.shape[1:],n_class=nb_classes,routings=routings,modeltype=modeltype)
        
        if "crossentropy" in modeltype:
            #model.compile(optimizer=optimizers.Adam(lr=lr,epsilon=1e-08),loss=['binary_crossentropy', 'mse'],loss_weights=[1., lam_recon],metrics={'capsnet': 'accuracy'})
            model.compile(optimizer=optimizers.Adam(lr=lr,epsilon=1e-08),loss='binary_crossentropy',metrics=['accuracy'])
            
        else:
            #model.compile(optimizer=optimizers.Adam(lr=lr,epsilon=1e-08),loss=[margin_loss, 'mse'],loss_weights=[1., lam_recon],metrics={'capsnet': 'accuracy'})
            #if Radam_flag:
            #    model.compile(optimizer=RAdam(),loss=margin_loss,metrics=['accuracy'])
            #else:
            model.compile(optimizer=optimizers.Adam(lr=lr,epsilon=1e-08),loss=margin_loss,metrics=['accuracy'])
        
    else: 
        model=compilemodels
    
    if(predict is False):
        if(weights is not None and compiletimes==0):
             print ("load weights:"+weights)
             model.load_weights(weights)
        
        print("##################save_best_only "+str(save_best_only))
        weight_checkpointer = LossModelCheckpoint(
             model_file_path =outputweights+'_iteration'+str(compiletimes),
             monitor_file_path= monitor_file +'_iteration' + str(compiletimes)+'.json',
             verbose=1,save_best_only=save_best_only,monitor='val_loss',mode='min',save_weights_only=True)
        
        if valX is not None:
            if(earlystop is None):
                 history=model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epoch,validation_data=[valX, valY],class_weight=class_weight,callbacks=[lr_decay,weight_checkpointer])
            else:
                 history=model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epoch,validation_data=[valX,valY],callbacks=[early_stopping,lr_decay,weight_checkpointer],class_weight=class_weight)
        else:
            history=model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epoch,class_weight=class_weight,callbacks=[lr_decay,weight_checkpointer])
        
        if load_average_weight:
            if save_best_only:
               last_weights =model.get_weights()
               model.load_weights(outputweights+'_iteration'+str(compiletimes)) #every iteration need to reload the best model for next run
               saved_weights = model.get_weights()
               avg_merged_weights = list()
               for layer in range(len(last_weights)):
                   avg_merged_weights.append(1/2*(last_weights[layer]+saved_weights[layer]))
               
               model.set_weights(avg_merged_weights)
        else:
            model.load_weights(outputweights+'_iteration'+str(compiletimes)) #every iteration need to reload the best model for next run
        
        #if load_average_weight:
        #    if save_best_only:
        #       last_weights =model.get_weights()
        #       model.load_weights(outputweights+'_iteration'+str(compiletimes)) #every iteration need to reload the best model for next run
        #       saved_weights = model.get_weights()
        #       avg_merged_weights = 1/2* Add()([last_weights,saved_weights]) #don't knwow if work
        #       model.set_weights(avg_merged_weights)
        #else:
        #    model.load_weights(outputweights+'_iteration'+str(compiletimes)) #every iteration need to reload the best model for next run
    
    return model

