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
from ex_7_model import *


if NEED_TRAIN:
    all_preds = []
    clean_train_df['pred'] = -1
    for i in range(N_FOLDS):
        print(f'Fold {i} results')
        learn, splitter = get_learner(fold_num=i)
        learn.fit_one_cycle(5, 2e-5, cbs=[SaveModelCallback(),
                                          EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=2)])
        learn.recorder.plot_loss()

        # over fitting
        learn.unfreeze()

        learn.fit_one_cycle(5, lr_max=slice(1e-6, 1e-4))

        learn = learn.to_fp32()

        learn.export(f'model_fold_{i}.pkl')

        learn.save(f'model_fold_{i}.pkl')

        dls = DataBlock(blocks=(ImageBlock, RegressionBlock),
                        get_x=ColReader('path'),
                        get_y=ColReader('norm_score'),
                        splitter=RandomSplitter(0.2),
                        item_tfms=Resize(224),  # pass in item_tfms
                        batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])
                        )
        paw_dls = dls.dataloaders(clean_train_df,
                                  bs=BATCH_SIZE,
                                  num_workers=8,
                                  seed=SEED
                                  )

        paw_dls = dls.dataloaders(train_df, bs=BATCH_SIZE, num_workers=8, seed=seed)

        test_dl = paw_dls.test_dl(test_df)

        preds, _ = learn.tta(dl=test_dl, n=5, beta=0)

        all_preds.append(preds)

        val_idx = splitter(range(len(clean_train_df)))[1]
        val_df = clean_train_df.loc[val_idx]
        val_pred, _ = learn.tta(dl=paw_dls.test_dl(val_df), n=5, beta=0)
        print(val_df['Pawpularity'][:5], val_pred[:5])
        score = mean_squared_error(val_df['Pawpularity'], val_pred * 100, squared=False)
        print(f'Fold {i} | Score: {score}')
        clean_train_df.loc[val_idx, 'pred'] = val_pred * 100
        del learn
        torch.cuda.empty_cache()
        gc.collect()
        if len(test_df) == 8:
            break

    if len(test_df) == 8:
        cv_score = mean_squared_error(clean_train_df.loc[clean_train_df['pred'] != -1, 'Pawpularity'],
                                      clean_train_df.loc[clean_train_df['pred'] != -1, 'pred'], squared=False)
        print(f'CV Score: {cv_score}')

if NEED_TRAIN:
    all_preds, np.mean(np.stack(all_preds * 100))


sample_df = pd.read_csv(dataset_path/'sample_submission.csv')

if NEED_TRAIN:
    preds = np.mean(np.stack(all_preds), axis=0)
    sample_df['Pawpularity'] = preds * 100
sample_df.to_csv('submission.csv', index=False)

if not NEED_TRAIN:
    pd.read_csv('submission.csv').head()
