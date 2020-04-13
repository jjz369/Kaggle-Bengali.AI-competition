# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def CustomSGDRScheduler(model, epoch, 
                        n_epochs = 200,
                        min_lr = 1e-7, 
                        max_lr = 0.001, 
                        lr_decay = 0.75, 
                        cycle_length = 15, 
                        multi_factor = 2):
    current_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
    print('Current LR: {0}'.format(current_lr))

    if multi_factor == 1:
        k = np.int(np.ceil(n_epochs/cycle_length))
    elif multi_factor > 1:
        k = np.int(np.ceil(np.log(n_epochs/cycle_length * (multi_factor - 1) +1) / np.log(multi_factor) ))

    epoch_range_list = []

    for i in range(k):
        epoch_range_list.append(np.ceil(((multi_factor)**i) * cycle_length ))
    epoch_range_cum = np.cumsum(epoch_range_list)

    idx = np.searchsorted(epoch_range_cum, epoch, side = 'left')
    if idx > 0:
        batch_since_restart = epoch - epoch_range_cum[idx-1]
    else:
        batch_since_restart = epoch

    restart_cycle_length = epoch_range_list[idx]
    restart_max_lr = max_lr * ((lr_decay)**idx)
  
    fraction_to_restart = batch_since_restart / restart_cycle_length
    new_lr = min_lr + 0.5*(restart_max_lr - min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
    tf.keras.backend.set_value(model.optimizer.lr, new_lr)

    print('index: {0}'.format(idx))
    print('Setting learning rate to: {0}'.format(new_lr))