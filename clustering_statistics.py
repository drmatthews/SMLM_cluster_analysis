#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from cluster import utils

import warnings
warnings.filterwarnings("ignore")


def collate_dataset_stats(out_dir, dataset_name, dataset_type, skip=[None]):
    print(dataset_name)
    df_list = []
    for filename in os.listdir(out_dir):
        if filename.endswith('xlsx') and (filename not in skip):
            if ((dataset_type in filename) and
                    (dataset_name in filename.lower())):

                df_list.append(
                    pd.read_excel(
                        os.path.join(out_dir, filename),
                        sheet_name='cluster statistics'
                    )
                )

    return pd.concat(df_list)


def cluster_near_neighbours(out_dir, dataset_name, dataset_type):
    print(dataset_name)
    nn_list = []
    for filename in os.listdir(out_dir):
        if filename.endswith('xlsx'):
            if (dataset_type in filename) and (dataset_name in filename):
                print(filename)
                df = pd.read_excel(
                    os.path.join(out_dir, filename),
                    sheet_name='cluster coordinates')

                nn = utils.kdtree_nn(
                    df.as_matrix(columns=['x [nm]', 'y [nm]']))
                nn_list.append(nn)

    nn_df = pd.DataFrame(np.concatenate(nn_list, axis=0))
    nn_df.columns = ['Distance [nm]']

    return nn_df


def mean_per_image(folder, dataset_name, sheet_name, parameter):
    mean = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(
                os.path.join(folder, filename),
                sheet_name=sheet_name)
            mean.append(df[parameter].mean())
    return np.array(mean)


def mean_near_neighbour_distance(folder, dataset_name, sheet_name):
    mean = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(
                os.path.join(folder, filename),
                sheet_name=sheet_name)
            cluster_mean = []
            for cid, c in df.groupby('cluster_id'):
                cluster_mean.append(c['nn distance [nm]'].mean())
            mean.append(sum(cluster_mean) / len(cluster_mean))
    return np.array(mean)
