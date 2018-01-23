from collections import defaultdict
from src.ExperimentMappings \
    import model_lookup, \
    loss_function_lookup, \
    optimizer_lookup, \
    scheduler_lookup, \
    data_source_delegates_lookup, \
    trainer_delegates_lookup, \
    evaluation_delegates_lookup, \
    saver_delegates_lookup
from src.Experiment import ExperimentFactory
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import logging
import xgboost as xgb
import torch
from multiprocessing import Pool
from tqdm import tqdm
import gc
import numpy as np # linear algebra
import pandas as pd # data processing
from math import isnan
from itertools import combinations
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel


def img_to_stats(paths):
    img_id, img = paths[0], paths[1]

    # ignored error
    np.seterr(divide='ignore', invalid='ignore')

    bins = 20
    scl_min, scl_max = -50, 50
    opt_poly = True
    # opt_poly = False

    try:
        st = []
        st_interv = []
        hist_interv = []
        for i in range(img.shape[2]):
            # Get a single channel for the image
            img_sub = np.squeeze(img[:, :, i])

            # median, max and min
            sub_st = []
            sub_st += [np.mean(img_sub), np.std(img_sub), np.max(img_sub), np.median(img_sub), np.min(img_sub)]
            # sub_st[1] = std
            # sub_st[2] = max
            # sub_st[3] = median
            # sub_st[4] = min
            # (sub_st[2] - sub_st[3]) = max - median also sub_st[-3] below
            # (sub_st[2] - sub_st[4]) = max - min also sub_st[-2] below
            # (sub_st[3] - sub_st[3]) = median - min also sub_st[-1] below
            sub_st += [(sub_st[2] - sub_st[3]), (sub_st[2] - sub_st[4]), (sub_st[3] - sub_st[4])]
            # sub_st[-3] =
            sub_st += [(sub_st[-3] / sub_st[1]), (sub_st[-2] / sub_st[1]),
                       (sub_st[-1] / sub_st[1])]  # normalized by stdev
            st += sub_st

            # Laplacian, Sobel, kurtosis and skewness
            st_trans = []
            st_trans += [laplace(img_sub, mode='reflect', cval=0.0).ravel().var()]  # blurr
            sobel0 = sobel(img_sub, axis=0, mode='reflect', cval=0.0).ravel().var()
            sobel1 = sobel(img_sub, axis=1, mode='reflect', cval=0.0).ravel().var()
            st_trans += [sobel0, sobel1]
            st_trans += [kurtosis(img_sub.ravel()), skew(img_sub.ravel())]

            if opt_poly:
                st_interv.append(sub_st)
                #
                st += [x * y for x, y in combinations(st_trans, 2)]
                st += [x + y for x, y in combinations(st_trans, 2)]
                st += [x - y for x, y in combinations(st_trans, 2)]

                # hist = list(cv2.calcHist([img], [i], None, [bins], [0., 1.]).flatten())
            hist = list(np.histogram(img_sub, bins=bins, range=(scl_min, scl_max))[0])
            hist_interv.append(hist)
            st += hist
            st += [hist.index(max(hist))]  # only the smallest index w/ max value would be incl
            st += [np.std(hist), np.max(hist), np.median(hist), (np.max(hist) - np.median(hist))]

        if opt_poly:
            for x, y in combinations(st_interv, 2):
                st += [float(x[j]) * float(y[j]) for j in range(len(st_interv[0]))]

            for x, y in combinations(hist_interv, 2):
                hist_diff = [x[j] * y[j] for j in range(len(hist_interv[0]))]
                st += [hist_diff.index(max(hist_diff))]  # only the smallest index w/ max value would be incl
                st += [np.std(hist_diff), np.max(hist_diff), np.median(hist_diff),
                       (np.max(hist_diff) - np.median(hist_diff))]

        # correction
        nan = -999
        for i in range(len(st)):
            if isnan(st[i]) == True:
                st[i] = nan

    except:
        print('except: ')

    return [img_id, st]

def extract_img_stats(paths):
    imf_d = {}
    p = Pool(8) #(cpu_count())
    ret = p.map(img_to_stats, paths)
    for i in tqdm(range(len(ret)), miniters=100):
        imf_d[ret[i][0]] = ret[i][1]

    ret = []
    fdata = [imf_d[i] for i, j in paths]
    return np.array(fdata, dtype=np.float32)

def process(df, bands):

    data = extract_img_stats([(k, v) for k, v in zip(df['id'].tolist(), bands)]); gc.collect()
    data = np.concatenate([data, df['inc_angle'].values[:, np.newaxis]], axis=-1); gc.collect()

    print(data.shape)
    return data


def main():
    # Do a random split of our training set into 70/15/15
    train_file = open("data/train.json")
    train = pd.read_json(train_file)
    train, test = train_test_split(train, test_size=0.15, stratify=train['is_iceberg'])

    tmp_path = Path("tmp/")
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_path = f"{tmp_path}/train.json"
    test_path = f"{tmp_path}/test.json"
    train.to_json(train_path)
    test.to_json(test_path)

    # Read in experiment_config
    experiment_config = json.load(open(opts.experiment_config_path, "r"))
    experiment_config["data_source_delegate"]["parameters"].update(
        {"testing_data_path": test_path,
         "training_data_path": train_path}
    )

    experiment_factory = ExperimentFactory(model_lookup,
                                           loss_function_lookup,
                                           optimizer_lookup,
                                           scheduler_lookup,
                                           data_source_delegates_lookup,
                                           trainer_delegates_lookup,
                                           evaluation_delegates_lookup,
                                           saver_delegates_lookup)

    experiment = experiment_factory.create_experiment(experiment_config, opts.study_save_path)
    for fold_num, data_fold in enumerate(experiment.data_source_delegate.retrieve_dataset_for_train()):

        train_df_data = defaultdict(list)
        val_df_data = defaultdict(list)
        for d in data_fold['train']:
            for k, v in d.items():
                if isinstance(v, torch.FloatTensor):
                    v = v.numpy()
                train_df_data[k].extend(v)
        for d in data_fold['val']:
            for k, v in d.items():
                if isinstance(v, torch.FloatTensor):
                    v = v.numpy()
                val_df_data[k].extend(v)

        train_df = pd.DataFrame(train_df_data)
        val_df = pd.DataFrame(val_df_data)

        train_X = process(train_df, np.transpose(np.stack(train_df['input'].values), (0, 3, 2, 1)))
        val_X = process(val_df, np.transpose(np.stack(val_df['input'].values), (0, 3, 2, 1)))
        #
        x1, x2 = train_X, val_X
        y1, y2 = train_df['label'], val_df['label']
        # First we train the GBM
        print('splitted: {0}, {1}'.format(x1.shape, x2.shape), flush=True)

        # XGB
        xgb_train = xgb.DMatrix(x1, y1)
        xgb_valid = xgb.DMatrix(x2, y2)
        watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]
        params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
                  'objective': 'binary:logistic', 'seed': 99, 'silent': True}
        params['eta'] = 0.03  # weight strinkage factor
        params['max_depth'] = 4
        params['subsample'] = 0.9  # ratio of features to be used
        params['eval_metric'] = 'logloss'
        params['colsample_bytree'] = 0.8
        params['colsample_bylevel'] = 0.8
        params['max_delta_step'] = 3
        # params['gamma'] = 5.0
        # params['labmda'] = 1
        params['scale_pos_weight'] = 1.0
        # params['seed'] = split_seed + r
        nr_round = 2000
        min_round = 100

        model1 = xgb.train(params,
                           xgb_train,
                           nr_round,
                           watchlist,
                           verbose_eval=50,
                           early_stopping_rounds=min_round)

        pred_xgb = model1.predict(xgb_valid, ntree_limit=model1.best_ntree_limit + 45)


        # Now train the neural network
        experiment.model, experiment.optimizer, experiment.scheduler = experiment.model_factory.create_model()
        for epoch in range(experiment.n_epochs):
            train, val = experiment.trainer_delegate.on_epoch_start(data_fold)
            for data in tqdm(train):
                experiment.model.train()
                model_output = experiment.trainer_delegate.create_model_output(data, experiment.model)
                transformed_output = experiment.trainer_delegate.apply_output_transformation(model_output)
                labels = experiment.trainer_delegate.create_data_labels(data)
                model_loss = experiment.trainer_delegate.calculate_loss(experiment.loss_function, transformed_output, labels)
                experiment.saver_delegate.save_results(data, transformed_output, model_loss, mode="TRAIN")
                experiment.trainer_delegate.apply_backward_pass(experiment.optimizer, model_loss)

            for data in tqdm(val):
                experiment.model.eval()
                model_output = experiment.trainer_delegate.create_model_output(data, experiment.model)
                transformed_output = experiment.trainer_delegate.apply_output_transformation(model_output)
                labels = experiment.trainer_delegate.create_data_labels(data)
                model_loss = experiment.trainer_delegate.calculate_loss(experiment.loss_function, transformed_output, labels)
                experiment.saver_delegate.save_results(data, transformed_output, model_loss, mode="VALIDATION")
            experiment.trainer_delegate.update_scheduler(experiment.saver_delegate.last_epoch_validation_loss,
                                                   experiment.scheduler)
            experiment.saver_delegate.update_all_results(fold_num, epoch)
            experiment.saver_delegate.update_loss_results(fold_num, epoch)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_save_path", default="../study_results/iceresnet_experiment")
    parser.add_argument("--experiment_config_path", default="study_configs/iceresnet_experiment.json")
    # parser.add_argument("--experiment_config_path", default="study_configs/triple_column_iceresnet_experiment.json")
    # parser.add_argument("--experiment_config_path", default="study_configs/iceresnet2_experiment.json")
    parser.add_argument("--logging_level", default="INFO")
    opts = parser.parse_args()
    log_level = logging.getLevelName(opts.logging_level)
    logging.basicConfig(level=log_level)
    main()
