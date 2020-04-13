# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_rand_bbox(width, height, l):
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    return r_x, r_y, r_l, r_w, r_h

class MultiOutputDataGenerator(ImageDataGenerator):
    # custom image generator
    def __init__(self, featurewise_center = False, samplewise_center = False, 
                 featurewise_std_normalization = False, samplewise_std_normalization = False, 
                 zca_whitening = False, zca_epsilon = 1e-06, rotation_range = 0.0, width_shift_range = 0.0, 
                 height_shift_range = 0.0, brightness_range = None, shear_range = 0.0, zoom_range = 0.0, 
                 channel_shift_range = 0.0, fill_mode = 'nearest', cval = 0.0, horizontal_flip = False, 
                 vertical_flip = False, rescale = None, preprocessing_function = None, data_format = None, validation_split = 0.0, 
                 mix_up_alpha = 0.0, cutmix_alpha = 0.0): # additional class argument
    
        # parent's constructor
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, 
                         zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, 
                         shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, 
                         preprocessing_function, data_format, validation_split)


       # Mix-up
        assert mix_up_alpha >= 0.0
        self.mix_up_alpha = mix_up_alpha
        
        # Cutmix
        assert cutmix_alpha >= 0.0
        self.cutmix_alpha = cutmix_alpha

    def mix_up(self, X1, y1, X2, y2, y_column_names):
        assert X1.shape[0] == X2.shape[0]
        lam = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
        X = X1 * lam + X2 * (1-lam)

        y = {}
        for output in y_column_names:
          y[output] = y1[output]*lam + y2[output]*(1-lam)

        return X, y
    
    def cutmix_shade(self, X1, y1, X2, y2):
        assert X1.shape[0] == X2.shape[0]
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        width = X1.shape[2]
        height = X1.shape[1]
        r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)
        bx1 = np.clip(r_x - r_w // 2, 0, width)
        by1 = np.clip(r_y - r_h // 2, 0, height)
        bx2 = np.clip(r_x + r_w // 2, 0, width)
        by2 = np.clip(r_y + r_h // 2, 0, height)
        X1[:, bx1:bx2, by1:by2, :] = X1[:, bx1:bx2, by1:by2, :]*max(lam, 1-lam)
        X = X1

        return X, y1

    def cutmix(self, X1, y1, X2, y2, y_column_names):
        assert X1.shape[0] == X2.shape[0]
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        width = X1.shape[2]
        height = X1.shape[1]
        r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)
        bx1 = np.clip(r_x - r_w // 2, 0, width)
        by1 = np.clip(r_y - r_h // 2, 0, height)
        bx2 = np.clip(r_x + r_w // 2, 0, width)
        by2 = np.clip(r_y + r_h // 2, 0, height)
        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
        X = X1
        lam_label = 1-(bx2-bx1)*(by2-by1)/(width*height)
        y = {}
        for output in y_column_names:
          y[output] = y1[output]*lam_label + y2[output]*(1-lam_label)
        return X, y

    

    def flow_from_dataframe(self, df, directory = None, x_col = 'filename',
                            y_col = 'class', y_column_names = None,
                            weight_col=None, classes_num = None,
                            target_size =(128, 128), color_mode = 'grayscale',
                            classes=None, class_mode = 'raw', batch_size=32,
                            shuffle=True, seed=None, save_to_dir=None,
                            save_prefix='', save_format='png', subset=None,
                            interpolation='nearest', validate_filenames=True):
      
        batches = super().flow_from_dataframe(df, directory = directory,
                                              x_col = x_col, y_col = y_col,
                                              batch_size=batch_size,
                                              class_mode = class_mode,
                                              target_size = target_size,
                                              color_mode = color_mode)
        while True:
            batch_x, batch_y = next(batches)

            target_dict = {}
            i = 0
            for output in y_column_names:
                target_dict[output] = pd.get_dummies(batch_y[:,i]).reindex(
                                      columns=range(0, classes_num[i]), fill_value=0).values
                i += 1
            
            # mixup or cutmix
            if (self.mix_up_alpha > 0) or (self.cutmix_alpha > 0):
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break

                target_dict_2 = {}
                i = 0
                for output in y_column_names:
                    target_dict_2[output] = pd.get_dummies(batch_y_2[:,i]).reindex(
                                          columns=range(0, classes_num[i]), fill_value=0).values
                    i += 1

                if (self.mix_up_alpha > 0) and (self.cutmix_alpha > 0):
                    if np.random.rand() < 0.4:
                        batch_x, target_dict = self.mix_up(batch_x, target_dict, batch_x_2, target_dict_2, y_column_names)
                    elif np.random.rand() < 0.8:
                        batch_x, target_dict = self.cutmix(batch_x, target_dict, batch_x_2, target_dict_2, y_column_names)
                    elif np.random.rand() < 0.9:
                        batch_x, target_dict = self.cutmix_shade(batch_x, target_dict, batch_x_2, target_dict_2)
                    
                elif self.mix_up_alpha > 0:
                    batch_x, target_dict = self.mix_up(batch_x, target_dict, batch_x_2, target_dict_2, y_column_names)
                else:
                    batch_x, target_dict = self.cutmix_shade(batch_x, target_dict, batch_x_2, target_dict_2)
          

            yield batch_x, target_dict