"""
EDA
"""
import numpy as np
import pandas as pd
import sys
from timm import create_model
from fastai.vision.all import *
import matplotlib.pyplot as plt
import PIL
from ex3_utilities import *


len_df = len(train_df)  # check the number of images in the training set
print(f"There are {len_df} images")

plt.hist(train_df['Pawpularity'])  # show a histogram of the images and popularity
plt.show()

print(f"The mean Pawpularity score is {train_df['Pawpularity'].mean()}") # check the mean

print(f"The median Pawpularity score is {train_df['Pawpularity'].median()}")  # check the median

print(f"The standard deviation of the Pawpularity score is {train_df['Pawpularity'].std()}")  # check standard deviation

# the number of unique values of pawpularity score
print(f"There are {len(train_df['Pawpularity'].unique())} unique values of Pawpularity score")

train_df['norm_score'] = train_df['Pawpularity']/100  # normalize the pawpularity score
print(train_df['norm_score'])

# print an image to have a look
im = Image.open(train_df['path'][1])
width, height = im.size
print(width, height)
print(im)  # fix_me
