# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import time
import gc
import os
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import config
from model import build_model
from image_transform import MultiOutputDataGenerator
from scheduler import CustomSGDRScheduler


random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_random_seed(config.SEED)

def load_clean_data():
    # read the meta data and get rid of the images data with incorrect labels. 
    # see https://www.kaggle.com/c/bengaliai-cv19/discussion/123859
    # and https://www.kaggle.com/c/bengaliai-cv19/discussion/126833 for more information.
    df = pd.read_csv(os.path.join(config.DATA_DIR,'train.csv'))
    df['filename'] = df.image_id.apply(lambda filename: config.TRAIN_DIR+filename+'.png')
    df = df[(df['image_id']!='Train_49823') & (df['image_id']!='Train_2819') & (df['image_id']!='Train_20689')]
    df = df[(df['image_id']!='Train_532') & (df['image_id']!='Train_1482') & (df['image_id']!='Train_1874')]
    return df
    
def config_model(train_model, learning_rate = 0.0005):    

    train_model.compile(
        optimizer = Adam(lr = learning_rate), 
        loss = {'grapheme': 'categorical_crossentropy', 
                'vowel': 'categorical_crossentropy',
                'consonant': 'categorical_crossentropy'},
        loss_weights = {'grapheme': 0.7,
                        'vowel': 0.15,
                        'consonant': 0.15},
        metrics={'grapheme': ['accuracy', tf.keras.metrics.Recall()],
                  'vowel': ['accuracy', tf.keras.metrics.Recall()],
                  'consonant': ['accuracy', tf.keras.metrics.Recall()]}
        )
    
def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name, 
                            monitor = 'val_grapheme_acc', 
                            verbose = 1, 
                            save_best_only = False, 
                            save_weights_only = True, 
                            mode = 'max', 
                            save_freq = 'epoch')


def train(pretrained_model, scale_factor, n_epochs = 200, learning_rate = 0.01):
    
    df = load_clean_data()
         
    new_width = int(scale_factor * config.WIDTH)
    new_height = int(scale_factor * config.HEIGHT)
    
    train_model = build_model(pretrained_model, flag = "train", image_shape = (new_height, new_width, config.CHANNEL))
    print(train_model.summary())
    
    config_model(train_model, learning_rate)
    
    datagen_train = MultiOutputDataGenerator(
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        mix_up_alpha = 0.4,         
        cutmix_alpha = 0.4)         

    datagen_val = MultiOutputDataGenerator()
    
    csv_logger = CSVLogger(config.RUN_DIR+'log.csv', append=True, separator=';')
    
    msss = MultilabelStratifiedShuffleSplit(n_splits = n_epochs, test_size = config.TEST_SIZE, random_state = config.SEED)
    start_time = time.time()
    for epoch, msss_splits in zip(range(0, n_epochs), msss.split(df, df[['grapheme_root','vowel_diacritic', 'consonant_diacritic']])):

        train_idx = msss_splits[0]
        valid_idx = msss_splits[1]
        np.random.shuffle(train_idx)
    
        train_gen = datagen_train.flow_from_dataframe(df.iloc[train_idx], 
          x_col = 'filename', 
          class_mode = 'raw', 
          classes_num = [168, 11, 7], 
          y_col = ['grapheme_root','vowel_diacritic', 'consonant_diacritic'], 
          y_column_names = ['grapheme','vowel', 'consonant'], 
          target_size =(new_height, new_width), 
          batch_size = config.BATCH_SZ,
          color_mode ='grayscale') 
        
        val_gen = datagen_val.flow_from_dataframe(df.iloc[valid_idx], 
          x_col = 'filename', 
          class_mode = 'raw', 
          classes_num = [168, 11, 7], 
          y_col = ['grapheme_root','vowel_diacritic', 'consonant_diacritic'], 
          y_column_names = ['grapheme','vowel', 'consonant'], 
          target_size =(new_height, new_width),  
          batch_size = config.BATCH_SZ,
          color_mode ='grayscale') 
        
        train_steps = round(len(train_idx) /config.BATCH_SZ) + 1
        valid_steps = round(len(valid_idx) / config.BATCH_SZ) + 1
    
        train_model.fit_generator(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=1,
            validation_data = val_gen,
            validation_steps=valid_steps,
            callbacks=[csv_logger, ModelCheckpointFull(config.RUN_DIR + 'model_' + str(epoch) + '.h5')]
            )
    
        # Custom ReduceLRonPlateau
        CustomSGDRScheduler(train_model, epoch, min_lr = 1e-7, max_lr = learning_rate, lr_decay = 0.1, cycle_length = int(n_epochs/2), multi_factor = 1)
        print(time.time() - start_time)

        del train_gen, val_gen, train_idx, valid_idx
        gc.collect()    



def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--shape", type = float, 
                        help = "New image size scale factor in terms of original image size")
    
    parser.add_argument("-n", "--nepochs", type = int, 
                        help = "Number of epochs to train")
    
    parser.add_argument("-lr", "--lrate", type = float, 
                        help = "Initial learning rate")
    
    parser.add_argument("-m", "--model", type = str, choices=["B3", "Seresnext50"],
                        help = "The pretrained model")

    
    args = parser.parse_args()
    
    if args.shape:
        scale_factor = args.shape
    else:
        scale_factor = 1.0
        
   
    train(args.model, scale_factor, args.nepochs, args.lrate)

    

if __name__ == "__main__":
    main()