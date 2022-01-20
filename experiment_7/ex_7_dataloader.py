"""

"""
import sys
import pandas as pd
import numpy as np
from timm import create_model
from timm.data.mixup import Mixup
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold
import gc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import PIL

from ex_7_utilities import *


def get_data(fold):
    train_df_f = clean_train_df.copy()
    train_df_f['is_valid'] = (train_df_f['fold'] == fold)
    splitter = RandomSplitter(0.2)
    splitter = IndexSplitter(splitter(range(len(clean_train_df)))[1])

    dls = ImageDataLoaders.from_df(train_df,  # pass in train DataFrame
                                   valid_pct=0.2,  # 80-20 train-validation random split
                                   seed=2999,  # seed
                                   fn_col='path',  # filename/path is in the second column of the DataFrame
                                   label_col='norm_score',  # label is in the first column of the DataFrame
                                   y_block=RegressionBlock,  # The type of target
                                   bs=BATCH_SIZE,  # pass in batch size
                                   num_workers=8,
                                   item_tfms=Resize(224),
                                   batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()]))

    dls = DataBlock(blocks=(ImageBlock, RegressionBlock),
                    get_x=ColReader('path'),
                    get_y=ColReader('norm_score'),
                    splitter=splitter,
                    item_tfms=Resize(224),  # pass in item_tfms
                    batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])
                    )

    paw_dls = dls.dataloaders(train_df_f,
                              bs=BATCH_SIZE,
                              num_workers=8,
                              seed=SEED)

    return paw_dls, splitter
