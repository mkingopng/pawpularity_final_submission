"""
by: noOne
date: 211027
copied from a notebook written by 'ilovescience'
which can be found at https://www.kaggle.com/tanlikesmath/petfinder-pawpularity-eda-fastai-starter
"""
import numpy as np
import pandas as pd
import sys
import tqdm as tqdm
from timm import create_model
from fastai.vision.all import *
import matplotlib.pyplot as plt
import PIL
from ex3_utilities import *
from ex3_loader import *
from ex3_model import *

# In fastai, the trainer class is the `Learner`, which takes in the data, model, optimizer, loss function, etc. and
# allows you to train models, make predictions, etc. Let's define the `Learner` for this task, and also use mixed
# precision. NB: using `BCEWithLogitsLoss` to treat this as a classification problem.
learn = Learner(dls, model, loss_func=BCEWithLogitsLossFlat(), metrics=petfinder_rmse).to_fp16()

# In order to train a model, we need to find the most optimal learning rate, which can be achieved with fastai's
# learning rate finder:
learn.lr_find(end_lr=3e-2)

# fine-tune the model with the desired learning rate of 2e-5. Save the best model and use the early stopping callback.
learn.fit_one_cycle(10, 2e-5, cbs=[SaveModelCallback(), EarlyStoppingCallback(monitor='petfinder_rmse',
                                                                              comp=np.less,
                                                                              patience=99)])

learn.recorder.plot_loss()  # plot the loss

learn = learn.to_fp32()  # put the model learn rate back to fp32

learn.save('fine-tuned')  # save the model

learn.export()  # export the checkpoint to use later (i.e. for an inference kernel):

