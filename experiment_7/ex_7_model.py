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
from ex_7_dataloader import *


def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))


def get_learner(fold_num):
    data, splitter = get_data(fold_num)
    model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=data.c)
    learn = Learner(data,
                    model,
                    loss_func=BCEWithLogitsLossFlat(),
                    metrics=petfinder_rmse,
                    cbs=[Mixup(0.2)]).to_fp16()
    return learn, splitter

