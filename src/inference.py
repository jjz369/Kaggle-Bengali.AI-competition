# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
import pandas as pd
import argparse
from math import ceil

import tensorflow as tf

import config
from preprocess import resize_image
from model import build_model

class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, batch_size = 16, img_size = (512, 512, 3), *args, **kwargs):
        self.X = X
        self.indices = np.arange(len(self.X))
        self.batch_size = batch_size
        self.img_size = img_size
                    
    def __len__(self):
        return int(ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(indices)
        return X
    
    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size))
        
        for i, index in enumerate(indices):
            image = self.X[index]
            image = resize_image(image, self.img_size[1], self.img_size[0])            
            image = np.stack((image,)*config.CHANNEL, axis=-1)
            image = image.reshape(-1, self.img_size[0], self.img_size[1], config.CHANNEL)            
            X[i,] = image        
        return X


def inference(B3 = None, Seresnext = None, scale_factor = 1.0):
    
    new_width = int(scale_factor * config.WIDTH)
    new_height = int(scale_factor * config.HEIGHT)
    
    if B3:
        inference_model_B3 = build_model(pretrained_model = "B3", flag = "inference", image_shape = (new_height, new_width, config.CHANNEL))
        inference_model_B3.load_weights(B3)
        
    if Seresnext:
        inference_model_Se= build_model(pretrained_model = "Seresnext50", flag = "inference", image_shape = (new_height, new_width, config.CHANNEL))
        inference_model_Se.load_weights(Seresnext)
    
    tgt_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic']
    
    # Create Predictions
    row_ids, targets = [], []
    
    # Loop through Test Parquet files (X)
    for i in range(0, 4):
        # Test Files Placeholder
        test_files = []
    
        # Read Parquet file
        df = pd.read_parquet(os.path.join(config.DATA_DIR, 'test_image_data_'+str(i)+'.parquet'))
        image_ids = df['image_id'].values 
        df = df.drop(['image_id'], axis = 1)
    
        # Loop over rows in Dataframe and generate images 
        X = []
        for image_id, index in zip(image_ids, range(df.shape[0])):
            test_files.append(image_id)
            X.append(df.loc[df.index[index]].values)
    
        # Data_Generator
        data_generator_test = TestDataGenerator(X, batch_size = config.TEST_BATCH_SZ, img_size = (new_height, new_width, config.CHANNEL))
            
        if B3:
            preds_B3 = inference_model_B3.predict_generator(data_generator_test, verbose = 1)
            len_data = len(preds_B3)

        if Seresnext:
            preds_Seresnext = inference_model_Se.predict_generator(data_generator_test, verbose = 1)
            len_data = len(preds_Seresnext)
        
        # Loop over Preds    
        for i, image_id in zip(range(len(test_files)), test_files):
            
            for subi, col in zip(range(len_data), tgt_cols):
                
                row_ids.append(str(image_id)+'_'+col)
                
                if B3 and Seresnext:               
                    sub_preds1 = preds_B3[subi]
                    sub_preds2 = preds_Seresnext[subi]              
                    sub_pred_value = np.argmax((sub_preds1[i] + sub_preds2[i] )  / 2.0)
                    
                elif B3:
                    sub_preds  = preds_B3[subi]
                    sub_pred_value = np.argmax((sub_preds[i] ))
                elif Seresnext:
                    sub_preds  = preds_Seresnext[subi]
                    sub_pred_value = np.argmax((sub_preds[i] ))
                    
                targets.append(sub_pred_value)                 
        
        del df
        gc.collect()    
        
    # Create and Save Submission File
    submit_df = pd.DataFrame({'row_id':row_ids,'target':targets}, columns = ['row_id','target'])
    submit_df.to_csv(os.path.join(config.OUTPUT_DIR, 'submission.csv'), index = False)


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--shape", type = float, 
                        help = "New image size scale factor in terms of original image size")

    
    parser.add_argument("-b3", "--B3", type = str, 
                        help = "path to the B3 model weights")
    
    parser.add_argument("-se", "--Seresnext50", type = str, 
                    help = "path to the Seresnext50 model weights")

    
    args = parser.parse_args()
    
    if args.shape:
        scale_factor = args.shape
    else:
        scale_factor = 1.0
   
    if args.B3 and args.Seresnext50:
        print("Going to predict with an ensemble model of EfficientNet B3 and Seresnext50")
        inference(B3 = args.B3, Seresnext = args.Seresnext50, scale_factor = scale_factor)
    elif args.B3:
        print("Going to predict with EfficientNet B3 ")
        inference(B3 = args.B3, scale_factor = scale_factor)  

    elif args.Seresnext50:        
        print("Going to predict with Seresnext50")
        inference(Seresnext = args.Seresnext50, scale_factor = scale_factor)
         
    else:
        print("Error: No model specified!")

    

if __name__ == "__main__":
    main()

