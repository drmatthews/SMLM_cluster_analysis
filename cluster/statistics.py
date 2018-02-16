#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from .utils import *

import warnings
warnings.filterwarnings("ignore")


def collect_filenames(folder, dataset_name):
    fnames = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            fnames.append(filename)
    return fnames


def write_stats(output_path, data, dataset_name):
    if isinstance(data, dict):
        df = pd.DataFrame()
        for key, value in data.items():
            df[key] = value
    else:
        df = data

    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    if os.path.exists(output_path):
        book = load_workbook(output_path)
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    df.to_excel(writer, sheet_name=dataset_name, index=False)
    writer.save()


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

def percentage_of_total(folder,
                        sheet_name,
                        dataset_name,
                        condition='eq',
                        cluster_column='lk'):
    percent = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(os.path.join(folder, filename),
                               sheet_name=sheet_name)
            if 'eq' in condition:
                filtered = float(df[df[cluster_column] == -1].shape[0])
            elif 'gt' in condition:
                filtered = float(df[df[cluster_column] > -1].shape[0])
            print("number of filtered locs: {0}".format(filtered))
            total = float(df.shape[0])
            print("total number of localisations: {0}".format(total))
            p = filtered / total
            print("percentage: {0}".format(p))
            percent.append(p)
    return np.array(percent)


def percentage_objects_with_clusters(folder, sheet_name, dataset_name):
    percent = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(os.path.join(folder, filename),
                               sheet_name=sheet_name)

            objects = float(df[df['object_id'] > -1].shape[0])
            clusters = float(df[df['cluster_id'] > -1].shape[0])

            print("number of object locs: {0}".format(objects))
            print("number of cluster locs: {0}".format(clusters))
            p = objects / clusters
            print("percentage: {0}".format(p))
            percent.append(p)
    return np.array(percent)


def mean_per_image(folder, dataset_name, parameter, sheet_name):
    mean = []
    for filename in os.listdir(folder):
        if filename.endswith('xlsx') and (dataset_name in filename):
            print(filename)
            df = pd.read_excel(
                os.path.join(folder, filename),
                sheet_name=sheet_name)
            if 'labels' in df:
                df = df[df['labels'] > -1]
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
