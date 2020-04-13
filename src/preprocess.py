# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import cv2
import gc
import argparse
from tqdm.auto import tqdm

import config

def resize_image(img, new_width, new_height):
    """ preprocess images:
        1) change images to dark background
        2) normalize the image matrix
        3) resize the image with defined scale factor.
    """  
        
    img = 255 - img

    # Normalize
    img = (img *(255.0/img.max())).astype(np.uint8)

    # Reshape
    img = img.reshape(config.HEIGHT, config.WIDTH)
    image_resized = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)

    return image_resized


def resize_and_save_image(image_dir, img, new_width, new_height, image_id):
    """ Resize and save images in input/train or input/test folders.
    """
    # resize image
    image_resized = resize_image(img, new_width, new_height)

    # save image
    cv2.imwrite(image_dir + str(image_id) + '.png', image_resized)


def generate_images(flag, scale_factor):
    """ Preprocess the training/testing parquet files and generate training/testing 
        images as image_id.png files in input/train or input/test folders.
    """
    
    new_width = int(scale_factor * config.WIDTH)
    new_height = int(scale_factor * config.HEIGHT)
    
    if flag == "train":
        
        print("Preprocess the training parquet files")
        if os.path.isdir(config.TRAIN_DIR):
            print("Train dir already exists!")
            pass
        
        else:
            os.mkdir(config.TRAIN_DIR)
            for i in tqdm(range(0, 4)):      
    
                df = pd.read_parquet(os.path.join(config.DATA_DIR, 'train_image_data_'+str(i)+'.parquet'))
                image_ids = df['image_id'].values
                df = df.drop(['image_id'], axis =1)
        
                for image_id, index in zip(image_ids, range(df.shape[0])):
                    resize_and_save_image(config.TRAIN_DIR, df.loc[df.index[index]].values, 
                                          new_width, new_height, image_id)
        
                # cleanup
                del df
                gc.collect()
                
    elif flag == "test":
        
        print("Preprocess the testing parquet files")
        if os.path.isdir(config.TEST_DIR):
            print("Test dir already exists!")
            pass
        
        else:
            os.mkdir(config.TEST_DIR)
            for i in tqdm(range(0, 4)):      
    
                df = pd.read_parquet(os.path.join(config.DATA_DIR, 'test_image_data_'+str(i)+'.parquet'))
                image_ids = df['image_id'].values
                df = df.drop(['image_id'], axis =1)
        
                for image_id, index in zip(image_ids, range(df.shape[0])):
                    resize_and_save_image(config.TEST_DIR, df.loc[df.index[index]].values, 
                                          new_width, new_height, image_id)
        
                # cleanup
                del df
                gc.collect()



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", 
                        help = "preprocess training images", action = 'store_true')
    parser.add_argument("-te", "--test",
                        help = "preprocess testing images", action = 'store_true')  
    
    parser.add_argument("-s", "--shape", type = float, 
                        help = "new image size scale factor in terms of original image size")

    
    args = parser.parse_args()
    
    if args.train:
        generate_images(flag = "train", scale_factor = args.shape)
        
    elif args.test:
        generate_images(flag = "test", scale_factor = args.shape)
    
    

if __name__ == "__main__":
    main()