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
from ex3_loader import *


# define the model
model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=dls.c)


# define the metric we will use
def petfinder_rmse(input, target):
    return 100*torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))
