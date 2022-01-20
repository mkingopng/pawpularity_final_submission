"""
by: noOne
date: 211027
copied from a notebook written by 'ilovescience'
which can be found at https://www.kaggle.com/tanlikesmath/petfinder-pawpularity-eda-fastai-starter
"""
import numpy as np
import pandas as pd
import sys
from timm import create_model
from fastai.vision.all import *
import matplotlib.pyplot as plt
import PIL
from ex3_utilities import *


dls = ImageDataLoaders.from_df(train_df,  # pass in train DataFrame
                               valid_pct=0.2,  # 80-20 train-validation random split
                               seed=999,  # seed
                               fn_col='path',  # filename/path is in the second column of the DataFrame
                               label_col='norm_score',  # label is in the first column of the DataFrame
                               y_block=RegressionBlock,  # The type of target
                               bs=8,  # pass in batch size
                               num_workers=8,
                               item_tfms=Resize(224),  # pass in item_tfms
                               batch_tfms=setup_aug_tfms([Brightness(),
                                                          Contrast(),
                                                          Hue(),
                                                          Saturation()])
                               )  # pass in batch_tfms

dls.show_batch()

