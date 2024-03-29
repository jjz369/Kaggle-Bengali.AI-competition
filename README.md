# Bengali.AI Handwritten Grapheme Classification

Classify the components of handwritten Bengali. This is a kaggle competition hosted from 2019-12-20 to 2020-03-16. Link to the competition page [Kaggle Bengali.AI](https://www.kaggle.com/c/bengaliai-cv19/overview).

This project won a bronze medal with a ranking of top 9% in the competition. 

Author: Jingjie Zhang (jjzhang369@gmail.com); Mingda Zhu (mdzhu.pku@gmail.com)

# Table of Contents

*   [Solution Overview](#solution-overview)
*   [Directory Layout](#directory-layout)
*   [Installation](#installation)
    *   [Download data](##download-data)
    *   [Install requirements](##install-requirements)
*   [Usage](#usage)
    *   [Training](#training)
    *   [Inference](#inference)
*   [Extending this](#extend)



# Solution Overview
-----

The solution is an ensemble model with backbone of [EfficientNet B3](https://github.com/qubvel/efficientnet) and [Seresnext50](https://github.com/qubvel/classification_models). 

<br>
<div align="center">
  <img src="./solution_overview.jpg" width = "800", height = "500">
</div>

# Directory Layout
-----

```
.
|--- input
|      |--- train or test parquet and csv files # train and test files provided by the competition (https://www.kaggle.com/c/bengaliai-cv19/data)
|      |--- train                               # folder for preprocessed training image files
|      |--- test                                # folder for preprocessed testing image files 
|--- output                                     # folder for saving the submission.csv 
|--- model                                      # folder for trained weights 
└--- src                                        # scripts
```

# Installation 
-----

## Download data
* Clone this repo to your local computer 
* Download the complete datasets from the kaggle Bengali.AI competition page [here](https://www.kaggle.com/c/bengaliai-cv19/data) and unzip them in the ```./input``` folder. 
* You can use the [kaggle API](https://github.com/Kaggle/kaggle-api): ```kaggle competitions download -c 'bengaliai-cv19'```
* The testing datasets are provided for inference/test purposes. 


## Install requirements
* Install the requirements using ```pip install -r requirements.txt```
* Make sure to use the python 3 version and tensorflow 1.x version. 
* A new virtual enviroment is recommended to have a clean package setup.
* Runing this project requires using GPU. We suggest using the [Google Colab](https://colab.research.google.com)

# Usage
-----

## Training

* **Preprocess** the image data. This module reads, resizes and normalizes images, and saves them to the `./input/train/` folder. 

  In the `./src` folder, run: 
  ~~~
  $python preprocess.py -tr -s 1.25 
  ~~~
  
  `-s 1.25` is a scale factor to scale the image from 137x236 to 171x295. It can be any meaningful float number. 
  
* **train** the model:
   
  In the `./src` folder, run:
  ~~~
  $python train.py -s 1.25 -n 200 -lr 0.001 -m "B3" 
  ~~~  
   
  - `-s 1.25` is a scale factor that scaled the image. Should be the same as the preprocess step. 
  - `-n 200` number of epochs to train the model. If not given, 200 will be used. 
  - `-lr 0.001` initial learning rate. The training process uses the [sgdr: Stochastic Gradient Descent](https://arxiv.org/pdf/1608.03983.pdf).
  - `-m "B3"` the backbone model. The options are "B3" and "Seresnext50". 
  
  The trained model weights will be saved in the `./model` folder and can be used in the **inference** step. 
  
## Inference

  In the `./src` folder, run:
  ~~~
  $python inference.py -s 1.25 -b3 "../model/model_B3.h5" 
  ~~~  
  
  - `-s 1.25` is a scale factor that to scale the image. Should be the same as the training step. 
  - `-b3 "../model/model_B3.h5"` to specify the saved weights for the B3 model. 
  - If you want to use Se-resnext50 models, use `-se "../model/model_Seresnext50.h5"` to specify the saved weights for the Se-resnext50 model. 
  - If both "-b3" and "-se" are given, an ensemble of these two models will be used. 
  
# Extending this
-----

To add more image augmentations, modifications can be made in the  `image_transform.py`; To make changes on the training models, edit `model.py`; 
To change the global parameters, edit `config.py`; To change the learning rate schedule, edit the `scheduler.py`. 


  
  



  
  


                         
