"""
notes:
    - It was noted in the chat that the seed bas a big impact on performance. Apparently this is a known issue with CV
    models. I don't really understand yet. Need to research more. However, using 2999 as the seed did have a noticable
    impact on model performance.

    - Swin-Transformers became the go-to model quite early in the competition. It was accepted that these models offered
    SOTA performance. These models are huge and training them consumes substantial compute resources. Training on
    Kaggle, even at a batch size of 1, would cause memory issues.

    - I didn't explore other models enough. I need to learn more rather than just taking other's word for it.

    - I found (like many others) that training on my local machine, saving the trained model and uploading for inference
    was more effective than trying to train on Kaggle. I tried a number of different formats for saving the model and
    checkpoints (pickle, tar.gz, JSON, h5, bin, YAML). I don't really understand the pros & cons of each file type. This
    is something I need to research.

    -
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

sys.path.append('pytorch-image-models-master')

dataset_path = Path('../data')  # original dataset
original_train_df = pd.read_csv(dataset_path/"train.csv")

clean_dataset_path = Path('../clean_data')  # duplicates removed
clean_train_df = pd.read_csv(clean_dataset_path/"train.csv")

# preprocess the images
original_train_df['path'] = original_train_df['Id'].map(lambda x: str(dataset_path / 'train' / x) + '.jpg')
original_train_df = original_train_df.drop(columns=['Id'])  # drop the 'Id' column
original_train_df = original_train_df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe

# preprocess the images
clean_train_df['path'] = clean_train_df['Id'].map(lambda x: str(clean_dataset_path / 'train' / x) + '.jpg')
clean_train_df = clean_train_df.drop(columns=['Id'])  # drop the 'Id' column
clean_train_df = clean_train_df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe

clean_train_df['norm_score'] = clean_train_df['Pawpularity'] / 100  # normalize the pawpularity score

# parameters
SEED = 2999
set_seed(SEED, reproducible=True)
BATCH_SIZE = 32
NEED_TRAIN = False
N_FOLDS = 7

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# Sturges' rule
num_bins = int(np.floor(1 + np.log2(len(clean_train_df))))
clean_train_df['bins'] = pd.cut(clean_train_df['norm_score'], bins=num_bins, labels=False)
clean_train_df['fold'] = -1

stratified_kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for i, (_, train_index) in enumerate(stratified_kfold.split(clean_train_df.index, clean_train_df['bins'])):
    clean_train_df.iloc[train_index, -1] = i

clean_train_df['fold'] = clean_train_df['fold'].astype('int')
clean_train_df.fold.value_counts().plot.bar()
clean_train_df[clean_train_df['fold'] == 0].head()
clean_train_df[clean_train_df['fold'] == 0]['bins'].value_counts()
clean_train_df[clean_train_df['fold'] == 1]['bins'].value_counts()

test_df = pd.read_csv('../data/test.csv')

if len(test_df) != 8:
    NEED_TRAIN = True

test_df['Pawpularity'] = [1] * len(test_df)
test_df['path'] = test_df['Id'].map(lambda x: str(dataset_path / 'test' / x) + '.jpg')
test_df = test_df.drop(columns=['Id'])
clean_train_df['norm_score'] = clean_train_df['Pawpularity'] / 100





