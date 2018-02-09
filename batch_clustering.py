#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from read_roi import read_roi_file
from read_roi import read_roi_zip

from localisations import Localisations
from cluster.voronoi_ import voronoi, voronoi_segmentation
from cluster.utils import (optics_clustering,
                           kdtree_nn,
                           plot_optics_clusters)

import warnings
warnings.filterwarnings("ignore")

def collate_dataset_stats(out_dir, dataset_name, dataset_type, sheetname):
    print(dataset_name)
    df_list = []
    for filename in os.listdir(out_dir):
        if filename.endswith('xlsx'):
            if (dataset_type in filename) and (dataset_name in filename):
                print(filename)
                df_list.append(pd.read_excel(
                    os.path.join(out_dir,filename),
                    sheetname=sheetname))

    return pd.concat(df_list)

def cluster_near_neighbours(out_dir, dataset_name, dataset_type):
    print(dataset_name)
    nn_list = []
    for filename in os.listdir(out_dir):
        if filename.endswith('xlsx'):
            if (dataset_type in filename) and (dataset_name in filename):
                print(filename)
                df = pd.read_excel(
                                os.path.join(out_dir,filename),
                                sheetname='cluster coordinates')
                nn = kdtree_nn(df.as_matrix(columns=['x [nm]', 'y [nm]']))
                nn_list.append(nn)

    nn_df = pd.DataFrame(np.concatenate(nn_list, axis=0))
    nn_df.columns = ['Distance [nm]']

    return nn_df

def run_optics(parameters, verbose=True, use_roi=False, pixel_size=16.0):

    # parameters
    eps = parameters['eps']
    eps_extract = parameters['eps_extract']
    min_samples = parameters['min_samples']
    input_dir = parameters['input_dir']
    output_dir = parameters['output_dir']
    data_source = parameters['data_source']

    for filename in os.listdir(input_dir):

        if 'nstorm' in data_source:
            ext = 'txt'
        elif 'thunderstorm' in data_source:
            ext = 'csv'

        file_ext = os.path.splitext(filename)[1]
        if ext in file_ext:
            print("optics clustering for file %s"%filename)
            filepath = os.path.join(input_dir,filename)
            basename = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, basename+'_optics.xlsx')

            data = Localisations(filepath, source=data_source)
            points = data.points
            xy = points.as_matrix(columns=['x','y'])

            if use_roi:
                roi_zip_path = os.path.join(input_dir, basename + '_roiset.zip')
                roi_file_path = os.path.join(input_dir, basename + '_roiset.roi')
                if os.path.exists(roi_zip_path):
                    rois = read_roi_zip(roi_zip_path)
                elif os.path.exists(roi_file_path):
                    rois = read_roi_file(roi_file_path)
                else:
                    raise ValueError(("No ImageJ roi file exists -"
                                     "you should put the file in the same directory as the data"))

                coords = {}
                for roi_id, roi in rois.items():
                    for k, v in roi.items():
                        if not isinstance(v, str):
                            roi[k] = float(v) * pixel_size

                    coords[roi_id] = xy[(xy[:, 0] > roi['left']) &
                                        (xy[:, 0] < roi['left'] + roi['width']) &
                                        (xy[:, 1] > roi['top']) &
                                        (xy[:, 1] < roi['top'] + roi['height'])]
            else:
                coords = dict([('image', xy)])

            (optics_clusters, noise) = optics_clustering(coords,
                                                         eps=eps,
                                                         eps_extract=eps_extract,
                                                         min_samples=min_samples)

            if verbose:
                if optics_clusters:
                    for roi_id in optics_clusters.keys():
                        plot_filename = os.path.join(output_dir, basename + '_{0}_clusters.png'.format(roi_id))
                        if noise:
                            noise_in_roi = noise[roi_id]
                        else:
                            noise_in_roi = None
                        plot_optics_clusters(optics_clusters[roi_id],
                                             noise=noise_in_roi,
                                             save=True,
                                             filename=plot_filename)

            for roi in optics_clusters.keys():

                oc = optics_clusters[roi]
                if oc.n_clusters > 0:
                    oc.save(output_path, sheetname=roi)
                else:
                    print("No clusters found in %s"%filename)
                    continue


def run_voronoi(parameters, verbose=True, use_roi=False, pixel_size=16.0):

    # parameters
    density_factor = parameters['density_factor']
    min_samples = parameters['min_samples']
    input_dir = parameters['input_dir']
    output_dir = parameters['output_dir']

    for filename in os.listdir(input_dir):

        if filename.endswith('csv'):
            print("voronoi clustering for file %s"%filename)
            filepath = os.path.join(input_dir,filename)
            basename = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, basename+'_voronoi.xlsx')

            locs = pd.read_csv(filepath)

            if use_roi:
                roi_zip_path = os.path.join(input_dir, basename + '_roiset.zip')
                roi_file_path = os.path.join(input_dir, basename + '_roiset.roi')
                if os.path.exists(roi_zip_path):
                    rois = read_roi_zip(roi_zip_path)
                elif os.path.exists(roi_file_path):
                    rois = read_roi_file(roi_file_path)
                else:
                    raise ValueError(("No ImageJ roi file exists -"
                                     "you should put the file in the same directory as the data"))

                coords = {}
                for roi_id, roi in rois.items():
                    for k, v in roi.items():
                        if not isinstance(v, str):
                            roi[k] = float(v) * pixel_size

                    coords[roi_id] = locs[(locs['x [nm]'] > roi['left']) &
                                          (locs['x [nm]'] < roi['left'] + roi['width']) &
                                          (locs['y [nm]'] > roi['top']) &
                                          (locs['y [nm]'] < roi['top'] + roi['height'])].reset_index(drop=True)
            else:
                coords = dict([('image', locs)])

            for roi_id in coords.keys():
                df = coords[roi_id]

                if len(df.index) == 0:
                    print("no coords in roi")
                    continue

                voronoi_df = voronoi(df, density_factor, min_samples, show_plot=verbose, verbose=verbose)

                cluster_locs_df = voronoi_df[voronoi_df['lk'] != -1]
                labels = cluster_locs_df['lk'].unique()
                cluster_stats = {}
                cluster_stats['area'] = []
                cluster_stats['occupancy'] = []
                for m in labels:
                    cluster = cluster_locs_df[cluster_locs_df.lk == m]
                    cluster_stats['area'].append(cluster['area'].sum())
                    cluster_stats['occupancy'].append(len(cluster.index))

                cluster_stats_df = pd.DataFrame(cluster_stats)
                if any(labels):
                    writer = pd.ExcelWriter(output_path, engine='openpyxl')
                    if os.path.exists(output_path):
                        book = load_workbook(output_path)
                        writer.book = book
                        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

                    cluster_locs_df.to_excel(
                        writer,
                        sheet_name='{0} localisations'.format(roi_id),
                        index=False
                    )
                    cluster_stats_df.to_excel(
                        writer,
                        sheet_name='{0} voronoi stats'.format(roi_id),
                        index=False
                    )
                    writer.save()
                else:
                    print("No clusters found in %s"%filename)
                    continue

def collect_stats(df, col):
    labels = df[col].unique()
    stats = {}
    stats['area'] = []
    stats['occupancy'] = []
    stats['labels'] = labels
    for m in labels:
        group = df[df[col] == m]
        stats['area'].append(group['area'].sum())
        stats['occupancy'].append(len(group.index))
    return stats

def run_voronoi_segmentation(parameters, use_roi=False, segment_rois=False):

    # parameters
    oms = parameters['object_min_samples']
    cdf = parameters['cluster_density_factor']
    cms = parameters['cluster_min_samples']
    input_dir = parameters['input_dir']
    output_dir = parameters['output_dir']

    for filename in os.listdir(input_dir):

        if filename.endswith('csv'):
            print("voronoi clustering for file {}".format(filename))
            locs_path = os.path.join(input_dir, filename)
            basename = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, basename+'_voronoi.xlsx')

            roi_path = None
            if use_roi:
                roi_zip_path = os.path.join(input_dir, basename + '_roiset.zip')
                roi_file_path = os.path.join(input_dir, basename + '_roiset.roi')
                if os.path.exists(roi_zip_path):
                    roi_path = roi_zip_path
                elif os.path.exists(roi_file_path):
                    roi_path = roi_file_path
                else:
                    raise ValueError(("No ImageJ roi file exists -"
                                     "you should put the file in the same directory"
                                     "as the data"))

            voronoi_clusters = voronoi_segmentation(locs_path,
                                                    roi_path=roi_path,
                                                    segment_rois=segment_rois,
                                                    object_min_samples=oms,
                                                    cluster_density_factor=cdf,
                                                    cluster_min_samples=cms,
                                                    show_plot=True)

            for vc_id, vc in voronoi_clusters.items():
                vdf = vc['locs']
                object_locs_df = vdf[vdf['object_id'] != -1]
                cluster_locs_df = vdf[vdf['cluster_id'] != -1]

                object_stats = collect_stats(object_locs_df, 'object_id')
                cluster_stats = collect_stats(object_locs_df, 'cluster_id')

                object_stats_df = pd.DataFrame(object_stats)
                cluster_stats_df = pd.DataFrame(cluster_stats)
                writer = pd.ExcelWriter(output_path, engine='openpyxl')
                if os.path.exists(output_path):
                    book = load_workbook(output_path)
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

                cluster_locs_df.to_excel(
                    writer,
                    sheet_name='{0} localisations'.format(vc_id),
                    index=False
                )
                object_stats_df.to_excel(
                    writer,
                    sheet_name='{0} object stats'.format(vc_id),
                    index=False
                )
                cluster_stats_df.to_excel(
                    writer,
                    sheet_name='{0} cluster stats'.format(vc_id),
                    index=False
                )
                writer.save()

if __name__=='__main__':
    parameters = {}
    parameters['pixel_size'] = 16.0 #nm
    parameters['object_min_samples'] = 3
    parameters['cluster_min_samples'] = 3
    parameters['cluster_density_factor'] = 20
    parameters['input_dir'] = 'C:\\Users\\Daniel\\Documents\\Image processing\\Penny\\atto647 dstorm pre-hawk\\batch_test'
    parameters['output_dir'] = 'C:\\Users\Daniel\\Documents\\Image processing\\Penny\\atto647 dstorm pre-hawk\\batch_test\\out'
    parameters['data_source'] = 'thunderstorm'
    run_voronoi_segmentation(parameters, use_roi=False, segment_rois=False)
