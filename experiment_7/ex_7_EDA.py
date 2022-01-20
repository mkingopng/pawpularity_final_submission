"""
basic EDA
"""
import numpy as np
import pandas as pd
import sys
from timm import create_model
from fastai.vision.all import *
import matplotlib.pyplot as plt
import PIL
from ex_7_utilities import *

# There are a number of duplicate images in the original dataset.
# Getting rid of them makes a difference to model performance

# original data
len_df = len(original_train_df)  # check the number of images in the original training set
print(f"There are {len_df} images in the original dataset")

plt.hist(original_train_df['Pawpularity'])  # histogram of the original images and pawpularity
plt.title("original data")
plt.show()

print(f"The mean Pawpularity score in the original df is {original_train_df['Pawpularity'].mean()}")
print(f"The median Pawpularity score in the original df is {original_train_df['Pawpularity'].median()}")
print(f"The standard deviation of the Pawpularity score in the original df is {original_train_df['Pawpularity'].std()}")

# the number of unique values of pawpularity score
print(f"There are {len(original_train_df['Pawpularity'].unique())} unique Pawpularity scores in the original df")

# clean data
len_df = len(clean_train_df)  # check the number of images in the cleaned training set
print(f"There are {len_df} images in the cleaned dataset")

plt.hist(clean_train_df['Pawpularity'])  # histogram of the cleaned images and pawpularity
plt.title("cleaned data")
plt.show()

print(f"The mean Pawpularity score in the cleaned df is {clean_train_df['Pawpularity'].mean()}")
print(f"The median Pawpularity score in the cleaned df is {clean_train_df['Pawpularity'].median()}")
print(f"The standard deviation of the Pawpularity score in the cleaned df is {clean_train_df['Pawpularity'].std()}")

# the number of unique values of pawpularity score
print(f"There are {len(clean_train_df['Pawpularity'].unique())} unique Pawpularity scores in the cleaned df")

print(f"the normalised score is {clean_train_df['norm_score']}")

im = Image.open(clean_train_df['path'][1])  # print a random image to have a look
width, height = im.size
print(f"the image width is {width}, \nthe image height is {height}")

imgplot = plt.imshow(im)
plt.title("sample image")
plt.show()
