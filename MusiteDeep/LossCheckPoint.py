from keras.callbacks import Callback
import numpy as np
import warnings
import json


class LossModelCheckpoint(Callback):
  """Save the model after every epoch.
  
  `filepath` can contain named formatting options,
  which will be filled the value of `epoch` and
  keys in `logs` (passed in `on_epoch_end`).
  
  For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
  then the model checkpoints will be saved with the epoch number and
  the validation loss in the filename.
  
  # Arguments
      model_filepath: string, path to save the model file.
      loss_filepath: string, path to save the loss file.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`,
          the latest best model according to
          the quantity monitored will not be overwritten.
      mode: one of {auto, min, max}.
          If `save_best_only=True`, the decision
          to overwrite the current save file is made
          based on either the maximization or the
          minimization of the monitored quantity. For `val_acc`,
          this should be `max`, for `val_loss` this should
          be `min`, etc. In `auto` mode, the direction is
          automatically inferred from the name of the monitored quantity.
      save_weights_only: if True, then only the model's weights will be
          saved (`model.save_weights(filepath)`), else the full model
          is saved (`model.save(filepath)`).
      period: Interval (number of epochs) between checkpoints.
  """
  
  def __init__(self, model_file_path, monitor_file_path, monitor='val_loss',
               verbose=0,
               save_best_only=False, save_weights_only=False,
               mode='auto', period=1):
    super(LossModelCheckpoint, self).__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.model_file_path = model_file_path
    self.monitor_file_path = monitor_file_path
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.period = period
    self.epochs_since_last_save = 0
    
    if mode not in ['auto', 'min', 'max']:
      warnings.warn('ModelCheckpoint mode %s is unknown, '
                    'fallback to auto mode.' % (mode),
                    RuntimeWarning)
      mode = 'auto'
    
    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf
  
  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epochs_since_last_save += 1
    if self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      modelfilepath = self.model_file_path.format(epoch=epoch + 1, **logs)
      #monitorfilepath = self.monitor_file_path.format(epoch=epoch + 1, **logs)
      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          warnings.warn('Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning)
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s, saving monitor value to %s'
                    % (epoch + 1, self.monitor, self.best,
                       current, modelfilepath, self.monitor_file_path))
            self.best = current
            if self.save_weights_only:
              self.model.save_weights(modelfilepath, overwrite=True)
            else:
              self.model.save(modelfilepath, overwrite=True)
            # save training details to json file
            details = {
              'loss': logs.get('loss'),
              'val_loss':logs.get('val_loss'),
              'acc':logs.get('acc'),
              'val_acc':logs.get('val_acc')
            }
            with open(self.monitor_file_path, 'w') as monitor_fp:
                json.dump(details,monitor_fp)
          
          else:
            if self.verbose > 0:
              print('\nEpoch %05d: %s did not improve from %0.5f' %
                    (epoch + 1, self.monitor, self.best))
      else:
        if self.verbose > 0:
          print('\nEpoch %05d: saving model to %s, saveing monitor value to '
                '%s' % (epoch + 1, modelfilepath, self.monitor_file_path))
        if self.save_weights_only:
          self.model.save_weights(modelfilepath, overwrite=True)
        else:
          self.model.save(modelfilepath, overwrite=True)
        # save monitor value to file
        details = {
          'loss': logs.get('loss'),
          'val_loss': logs.get('val_loss'),
          'acc': logs.get('acc'),
          'val_acc': logs.get('val_acc')
        }
        with open(self.monitor_file_path, 'w') as monitor_fp:
          json.dump(details, monitor_fp)
