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

set_seed(2999, reproducible=True)  # set the seed

dataset_path = Path('../data/')  # set the path for the data folder

train_df = pd.read_csv(dataset_path / 'train.csv')  # read the train csv
print(train_df.head())

train_df['path'] = train_df['Id'].map(lambda x: str(dataset_path / 'train' / x) + '.jpg')  # preprocess the images
train_df = train_df.drop(columns=['Id'])  # drop the 'Id' column
train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe
print(train_df.head())

len_df = len(train_df)  # check the number of images in the training set
print(f"There are {len_df} images")

plt.hist(train_df['Pawpularity'])  # show a histogram of the images and popularity
plt.show()

print(f"The mean Pawpularity score is {train_df['Pawpularity'].mean()}")  # check the mean

print(f"The median Pawpularity score is {train_df['Pawpularity'].median()}") # check the median


print(f"The standard deviation of the Pawpularity score is {train_df['Pawpularity'].std()}")  # check standard deviation

# the number of unique values of pawpularity score
print(f"There are {len(train_df['Pawpularity'].unique())} unique values of Pawpularity score")

train_df['norm_score'] = train_df['Pawpularity']/100  # normalize the pawpularity score
print(train_df['norm_score'])

im = Image.open(train_df['path'][1])  # print an image to have a look
width, height = im.size
print(width, height)
