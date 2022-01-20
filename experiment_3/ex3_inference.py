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
from ex3_model import *
from ex3_train import *

# preprocess the test CSV in the same way as the train CSV
test_df = pd.read_csv(dataset_path / 'test.csv')
print(test_df.head())

# the `dls.test_dl` function allows us to create test dataloader using the same pipeline defined earlier.
test_df['Pawpularity'] = [1] * len(test_df)
test_df['path'] = test_df['Id'].map(lambda x: str(dataset_path / 'test' / x) + '.jpg')
test_df = test_df.drop(columns=['Id'])
train_df['norm_score'] = train_df['Pawpularity'] / 100
test_dl = dls.test_dl(test_df)

# We can easily confirm that the test_dl is correct:
print(test_dl.show_batch())

# Now let's pass the dataloader to the model and get predictions. Here I am using 5x test-time augmentation which
# further improves model performance.
preds, _ = learn.tta(dl=test_dl, n=5, beta=0)

# Let's make a submission with these predictions!
sample_df = pd.read_csv(dataset_path/'sample_submission.csv')
sample_df['Pawpularity'] = preds.float().numpy()*100
sample_df.to_csv('submission.csv', index=False)
