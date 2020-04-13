# -*- coding: utf-8 -*-
import efficientnet.tfkeras as efn
from classification_models.tfkeras import Classifiers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_model(pretrained_model, flag, image_shape):
    
    print(pretrained_model, flag)
    
    if pretrained_model == "B3" and flag == "train":

        x_model = efn.EfficientNetB3(weights = 'imagenet', include_top = False, 
                        input_shape = (image_shape[0], image_shape[1], 3), pooling = None, classes=None)

    elif pretrained_model == "B3" and flag == "inference":
        
        print("B3", "inference")
        x_model = efn.EfficientNetB3(weights = None, include_top = False, 
                        input_shape = (image_shape[0], image_shape[1], 3), pooling = None, classes=None)
        
    elif pretrained_model == "Seresnext50" and flag == "train":
        
        Sesnext, preprocess_input = Classifiers.get('seresnext50')
        x_model = Sesnext(weights='imagenet', include_top = False, 
                        input_shape = (image_shape[0], image_shape[1], 3), pooling = None, classes=None)
        
    elif pretrained_model == "Seresnext50" and flag == "inference":
        
        Sesnext, preprocess_input = Classifiers.get('seresnext50')
        x_model = Sesnext(weights = None, include_top = False, 
                        input_shape = (image_shape[0], image_shape[1], 3), pooling = None, classes=None)
        
    x_in = Input(shape = image_shape)
    x = Conv2D(3, (3, 3), padding = 'same')(x_in)
    x = x_model(x)

    for layer in x_model.layers:
      layer.trainable = True


    x1 = GlobalAveragePooling2D()(x) 
    x1 = Dense(256, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.25)(x1)


    x2 = GlobalAveragePooling2D()(x) 
    x2 = Dense(256, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.25)(x2)

    #x3 = lambda_layer(x)
    x3 = GlobalAveragePooling2D()(x)   
    x3 = Dense(256, activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.25)(x3)
  
    out_grapheme = Dense(168, activation='softmax', name='grapheme')(x1)
    out_vowel = Dense(11, activation='softmax', name='vowel')(x2)
    out_consonant = Dense(7, activation='softmax', name='consonant')(x3)
    
    model = Model(inputs=x_in, outputs=[out_grapheme, out_vowel, out_consonant])
    
    return model