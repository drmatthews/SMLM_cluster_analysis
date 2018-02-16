#!/usr/bin/env python
"""
A Python implementation of some of the ideas in the SR-Tesseler paper.
Basically this does is calculate the area (in pixels) of the Voroni
region around a localization and stores that in the localizations fit
area field.

Note: This ignores the localization category.

Note: This will handle up to on the order of 1M localizations. Analysis
      of files with a lot more localizations than this will likely
      take a long time to analyze.

Hazen 09/16

This is a modified version of code written by Hazen Babcock from the
Zhuang lab at Harvard Medical School.

This has been modified to work on localisations saved in a Pandas
DataFrame. It is possible to either give it a full dataset
(not recommended) or the ID of an ROI saved into the localisations
class object (in which case the localisations must have been parsed
using the Localisations class in "data_reader.py").

Added function to calculate alpha shape.

Dan 03/17
"""
import os
import math
import numpy as np
from scipy.stats import norm
import pandas as pd
from matplotlib import pylab as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from read_roi import read_roi_zip
from read_roi import read_roi_file
from .utils import plot_voronoi_diagram, plot_cluster_polygons


def build_voronoi(locs_df,
                  num_locs=None,
                  first_rank=False,
                  max_area=None,
                  show_plot=False,
                  verbose=True):

    if not isinstance(locs_df, pd.DataFrame):
        raise ValueError("only works with Pandas DataFrames created from"
                         "localisations exported from Thunderstorm")

    locs_df['area'] = np.nan
    n_locs = len(locs_df.index)
    if num_locs:
        n_locs = num_locs
    points = locs_df.as_matrix(columns=['x [nm]', 'y [nm]'])
    vor = Voronoi(points)
    for i, region_index in enumerate(vor.point_region):

        vertices = []
        for vertex in vor.regions[region_index]:

            # I think these are edge regions?
            if (vertex == -1):
                vertices = []
                break

            vertices.append(vor.vertices[vertex])

        if (len(vertices) > 0):
            locs_df.at[i, 'area'] = Polygon(vertices).area

    if max_area:
        locs_df = locs_df[locs_df['area'] <= max_area]

    if verbose:
        print("Min density", 1.0 / locs_df['area'].min())
        print("Max density", 1.0 / locs_df['area'].max())
        print("Median density", 1.0 / locs_df['area'].median())

    # calculate neighbors
    neighbors_counts = np.zeros(n_locs)
    if first_rank:
        unused_neighbors, neighbors_counts = polygon_neighbors(vor, n_locs)

    # use neighbours to calculate density
    density = np.divide(
        np.add(1, neighbors_counts).astype(float), locs_df['area'].values
    )
    locs_df = locs_df.assign(density=pd.Series(density))

    vor_fig = None
    if show_plot:
        vor_fig = plot_voronoi_diagram(vor)

    return {'voronoi': vor, 'locs': locs_df, 'vor_fig': vor_fig}


def voronoi_clustering(vor_data,
                       density_factor,
                       min_size,
                       cluster_column='lk',
                       num_locs=None):

    if not isinstance(vor_data, dict):
        raise ValueError(("You must pass a dictionary containing voronoi"
                          "polygons and a Pandas DataFrame of localisations"))

    locs_df = vor_data['locs'].copy()
    vor = vor_data['voronoi']
    if 'area' in locs_df:
        n_locs = locs_df.index.max() + 1
        if num_locs:
            n_locs = num_locs
        print("n_locs in voronoi_clustering: {}".format(n_locs))
        ave_density = density(locs_df)
        print("ave density: {}".format(ave_density))
        if cluster_column not in locs_df:
            locs_df[cluster_column] = -1
        min_density = density_factor * ave_density
        visited = np.zeros((n_locs), dtype = np.int32)

        neighbors, neighbors_counts = polygon_neighbors(vor, n_locs)

        def neighborsList(index):
            nlist = []
            for i in range(neighbors_counts[index]):
                loc_index = neighbors[index, i]
                # check that the index is in the localisations DataFrame
                # it might be a filtered list
                if loc_index not in locs_df.index:
                    continue
                if (visited[loc_index] == 0):
                    nlist.append(neighbors[index, i])
                    visited[loc_index] = 1
            return nlist

        cluster_id = 2
        # loop over the localisations but use the index values of the DataFrame
        # because this might be a filtered list
        for i in locs_df.index.values:
            if (visited[i] == 0):
                # index the DataFrame using the index value
                if (locs_df.loc[i]['density'] > min_density):
                    cluster_elt = [i]
                    c_size = 1
                    visited[i] = 1
                    to_check = neighborsList(i)
                    while (len(to_check) > 0):

                        # Remove last localization from the list.
                        loc_index = to_check[-1]
                        to_check = to_check[:-1]

                        # If the localization has sufficient density add to cluster and check neighbors.
                        if (locs_df.loc[loc_index]['density'] > min_density):
                            to_check += neighborsList(loc_index)
                            cluster_elt.append(loc_index)
                            c_size += 1

                        # Mark as visited.
                        visited[loc_index] = 1

                    # Mark the cluster if there are enough localizations in the cluster.
                    if (c_size > min_size):
                        # print("cluster", cluster_id, "size", c_size)
                        for elt in cluster_elt:
                            locs_df.at[elt, cluster_column] = cluster_id
                        cluster_id += 1
                visited[i] = 1
        return {'voronoi': vor, 'locs': locs_df}
    else:
        raise ValueError("No polygons found - build voronoi diagram first")


def voronoi_montecarlo(roi,
                       iterations=100,
                       confidence=99,
                       pixel_size=16.0,
                       segment_clusters=False,
                       density_factor=2,
                       min_size=5,
                       show_plot=True,
                       verbose=False):

    vor = build_voronoi(roi['locs'], show_plot=show_plot, verbose=verbose)
    locs_df = vor['locs']

    z = norm.ppf(1 - float(100 - confidence) * 0.01 / 2.0)
    # the following will work as long as roi['type'] == 'rectangle'
    w = roi['width']
    h = roi['height']
    a =  w * h
    locs_density = density(locs_df)
    num_rand_locs = int(locs_density * a)
    bins = round(2 * len(roi['locs'].index)**(1/3))
    mc_counts = np.zeros((iterations, bins))
    for i in range(0, iterations):
        x = w * np.random.random((num_rand_locs, 1))
        y = h * np.random.random((num_rand_locs, 1))
        xy = pd.DataFrame(np.hstack((x, y)))
        xy.columns = ['x [nm]', 'y [nm]']
        rand_vor = build_voronoi(xy, show_plot=False, verbose=False)
        area = rand_vor['locs']['area'].values
        if i == 0:
            lim = 3 * np.median(area[~np.isnan(area)])

        c, e = np.histogram(area, bins=bins, range=(0, lim))
        mc_counts[i, :] = c[:]

    counts, edges = np.histogram(
            locs_df['area'].values, bins=bins, range=(0, lim))
    centers = centers = (edges[:-1] + edges[1:]) / 2
    mean_mc_counts = np.mean(mc_counts, axis=0)
    std_mc_counts = np.std(mc_counts, axis=0)
    upper_mc_counts = np.add(mean_mc_counts, z * std_mc_counts)
    lower_mc_counts = np.subtract(mean_mc_counts, z * std_mc_counts)

    ind = np.where((counts - mean_mc_counts) < 0.0)[0][0]
    x1 = np.array((centers[ind - 1], centers[ind]))
    y1 = np.array((counts[ind - 1], counts[ind]))
    x2 = x1
    y2 = np.array((mean_mc_counts[ind - 1], mean_mc_counts[ind]))

    inters = intersection(x1, y1, x2, y2)

    if segment_clusters:
        locs_df = voronoi_clustering(
                locs_df, density_factor, min_size, max_area=inters[0])

    if show_plot:
        plt.figure()
        plt.plot(centers, counts, 'k-')
        plt.plot(centers, mean_mc_counts, 'r-')
        plt.plot(centers, upper_mc_counts, 'b-')
        plt.plot(centers, lower_mc_counts, 'b-')
        plt.plot(inters[0], inters[1], 'g*')
        plt.show()

    roi['locs'] = locs_df
    roi['centers'] = centers
    roi['mc_mean'] = mean_mc_counts
    roi['mc_upper'] = upper_mc_counts
    roi['mc_lower'] = lower_mc_counts
    roi['intersection'] = inters[0]
    return roi


def voronoi(locs_path,
            roi_path=None,
            pixel_size=16.0,
            density_factor=2,
            min_size=5,
            show_plot=True,
            verbose=True):

    if locs_path.endswith('csv'): # Thunderstorm
        locs_df = pd.read_csv(locs_path)
        locs_density = (locs_df)

        if roi_path:
            if roi_path.endswith('zip'):
                rois = read_roi_zip(roi_path)
            elif roi_path.endswith('roi'):
                rois = read_roi_file(roi_path)
            else:
                raise ValueError(("No ImageJ roi file exists -"
                                 "you should put the file in the same directory"
                                 "as the data and make sure it has the same base"
                                 "filename as the localisations."))
        else:
            # use all the localisations but mimic the rois dict data structure
            rois = {'image': {'locs': locs_df, 'width': dx, 'height': dy}}

        for roi_id, roi in rois.items():
            vor = build_voronoi(rois[roi_id]['locs'], show_plot=show_plot, verbose=verbose)
            locs_df = voronoi_clustering(vor['locs'], density_factor, min_size)

            if show_plot:
                plot_voronoi_diagram(
                        vor['voronoi'], plot_clusters=True, locs_df=locs_df)

        return locs_df
    else:
        raise ValueError("This can only handle data from Thunderstorm at present")


def voronoi_segmentation(locs_path,
                         roi_path=None,
                         segment_rois=False,
                         pixel_size=16.0,
                         monte_carlo=True,
                         object_density_factor=2,
                         object_min_samples=3,
                         cluster_density_factor=20,
                         cluster_min_samples=3,
                         num_rois=5,
                         roi_size=7000.0,
                         show_plot=False):
    """
    This uses a Monte Carlo simulation to predict which
    localisations are no better than a random distribution.
    This is used to do a first pass segmentation of 'objects'
    after which 'clusters' are located in those in those objects
    based on local density and a minimum number of objects.
    """
    # read in the complete localisations
    if locs_path.endswith('csv'): # Thunderstorm
        locs = pd.read_csv(locs_path)
    else:
        raise ValueError("This can only handle data from Thunderstorm at present")

    if roi_path:
        if roi_path.endswith('zip'):
            input_rois = read_roi_zip(roi_path)
        elif roi_path.endswith('roi'):
            input_rois = read_roi_file(roi_path)
        else:
            raise ValueError(("No ImageJ roi file exists -"
                             "you should put the file in the same directory"
                             "as the data and make sure it has the same base"
                             "filename as the localisations."))
        # add the localisations to the rois dict
        rois = roi_coords(locs, input_rois, pixel_size)
    else:
        # generate a random set of rois
        rois = random_rois(locs, num_rois, roi_size)

    if monte_carlo:
        intersection = 0.0
        for roi_id, roi in rois.items():
            print("Monte Carlo simulation in ROI {0}".format(roi_id))
            # pass the locs DataFrame and roi dict to voronoi_montecarlo
            rmc = voronoi_montecarlo(roi, show_plot=False)
            intersection += rmc['intersection']

        # always use the average intersection
        intersection /= float(len(rois))
        thresh = 1.0 / intersection

    # density factor for each roi (whether random roi or IJ roi)
    density_factor = {}
    for roi_id, roi in rois.items():
        if monte_carlo:
            density_factor[roi_id] = thresh / density(roi['locs'])
        else:
            density_factor[roi_id] = object_density_factor

    if not segment_rois:
        # use all the localisations but mimic the rois dict data structure
        rois = {'image': {'locs': locs}}
        # and reset the density_factor for the whole image
        density_factor = {}
        if monte_carlo:
            density_factor['image'] = thresh / density(rois['image']['locs'])
        else:
            density_factor['image'] = object_density_factor

    print("density_factor: {}".format(density_factor))

    # now loop over the rois data structure
    for roi_id, roi in rois.items():
        # build the voronoi diagram
        roi_locs = roi['locs']
        roi_locs['object_id'] = -1
        roi_locs['cluster_id'] = -1
        vor = build_voronoi(roi_locs)

        if monte_carlo:
            print("intersection: {0}".format(intersection))
            print("density threshold: {0}".format(thresh))

        print("density_factor: {}".format(density_factor[roi_id]))

        objs = voronoi_clustering(vor,
                                  density_factor[roi_id],
                                  object_min_samples,
                                  cluster_column='object_id')

        # write the objects to the original localisations data structure
        obj_locs = objs['locs']
        roi_locs.loc[roi_locs.object_id.isin(obj_locs.object_id), ['object_id']] = (
            obj_locs[obj_locs['object_id'] > -1]['object_id'])

        # retain only the objects that are 'clustered'
        filtered = obj_locs[obj_locs['object_id'] != -1]

        # prepare the data structure for re-clustering
        data = {'voronoi': vor['voronoi'], 'locs': filtered}

        # set parameters and find clusters in objects
        n_locs = len(roi_locs.index)
        clusters = voronoi_clustering(data,
                                      cluster_density_factor,
                                      cluster_min_samples,
                                      cluster_column='cluster_id',
                                      num_locs=n_locs)

        # write the clusters to the original localisations data structure
        clust_locs = clusters['locs']
        roi_locs.loc[roi_locs.cluster_id.isin(clust_locs.cluster_id), ['cluster_id']] = (
            clust_locs[clust_locs['cluster_id'] > -1]['cluster_id'])

        if show_plot:
            cfig = plot_voronoi_diagram(clusters['voronoi'],
                                        locs_df=clusters['locs'],
                                        cluster_column='cluster_id')
            plot_cluster_polygons(objs['locs'],
                                  figure=cfig,
                                  cluster_column='object_id')
            plt.show()

        # fill nan values before returning
        roi_locs[['object_id', 'cluster_id']] = roi_locs[['object_id', 'cluster_id']].fillna(value=-1)
        roi['locs'] = roi_locs
    return rois
#
#  helpers
#
def roi_coords(locs, rois, pixel_size):
    for roi_id, roi in rois.items():
        for k, v in roi.items():
            if not isinstance(v, str):
                roi[k] = float(v) * pixel_size

        rois[roi_id]['locs'] = roi_locs(locs, roi)

    return rois


def roi_locs(locs, roi):
    return locs[(locs['x [nm]'] > roi['left']) &
                (locs['x [nm]'] < roi['left'] + roi['width']) &
                (locs['y [nm]'] > roi['top']) &
                (locs['y [nm]'] < roi['top'] + roi['height'])].reset_index(drop=True)


def polygon_neighbors(vor, n_locs):
    # Record the neighbors of each point.
    max_neighbors = 40
    neighbors = np.zeros((n_locs, max_neighbors), dtype = np.int32) - 1
    neighbors_counts = np.zeros((n_locs), dtype = np.int32)
    print("n_locs: {}".format(n_locs))
    for ridge_p in vor.ridge_points:

        p1 = ridge_p[0]
        p2 = ridge_p[1]

        # Add p2 to the list for p1
        neighbors[p1, neighbors_counts[p1]] = p2
        neighbors_counts[p1] += 1

        # Add p1 to the list for p2
        neighbors[p2, neighbors_counts[p2]] = p1
        neighbors_counts[p2] += 1

    if False:
        n1 = neighbors[0,:]
        print(n1)
        print(neighbors[n1[0],:])

    return (neighbors, neighbors_counts)

def fov(locs):
    x_min = locs['x [nm]'].min()
    x_max = locs['x [nm]'].max()
    y_min = locs['y [nm]'].min()
    y_max = locs['y [nm]'].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    fovx = math.ceil(x_range/1000.0) * 1000
    fovy = math.ceil(y_range/1000.0) * 1000
    return (fovx, fovy)

def density(locs):
    im_fov = max(fov(locs))
    n_locs = locs.index.max() + 1
    return n_locs / (im_fov * im_fov)

def intersection(x1, y1, x2, y2):
#      y = a*x + b
    a1 = (y1[1] - y1[0]) / (x1[1] - x1[0])
    a2 = (y2[1] - y2[0]) / (x2[1] - x2[0])
    b1 = y1[0] - a1 * x1[0]
    b2 = y2[0] - a2 * x2[0]
    xi = (b2 - b1) / (a1 - a2)
    yi = a1 * xi + b1
    return (xi, yi)

def random_rois(locs, num_rois=5, roi_size=7000.0):

    locs_fov = fov(locs)
    roi_fovx = locs_fov[0] - (roi_size / 2.0)
    # print("roi_fovx: {}".format(roi_fovx))
    roi_fovy = locs_fov[1] - (roi_size / 2.0)
    # print("roi_fovx: {}".format(roi_fovx))
    rand_roi_x = np.random.randint(low=(roi_size / 2.0), high=roi_fovx, size=num_rois)
    rand_roi_left = np.subtract(rand_roi_x, roi_size / 2.0)
    # print("rand_roi_x: {}".format(rand_roi_x))
    rand_roi_y = np.random.randint(low=(roi_size / 2.0), high=roi_fovy, size=num_rois)
    rand_roi_top = np.subtract(rand_roi_y, roi_size / 2.0)
    # print("rand_roi_y: {}".format(rand_roi_y))

    rand_rois = {}
    for r in range(num_rois):
        roi = {}
        roi['type'] = 'rectangle'
        roi['left'] = rand_roi_left[r]
        roi['top'] = rand_roi_top[r]
        roi['width'] = roi_size
        roi['height'] = roi_size
        roi['locs'] = roi_locs(locs, roi)
        rand_rois['roi_0{}'.format(r)] = roi

    return rand_rois


if __name__ == '__main__':
    root_dir = 'C:\\Users\\Daniel\\Documents\\Image processing\\Penny\\atto647 dstorm pre-hawk\\test'
    locs_filename = 'run03StormTableLeftB wtTNF_filtered.csv'
    locs_path = os.path.join(root_dir, locs_filename)
    voronoi_segmentation(locs_path)
    # mc = voronoi_montecarlo(locs_path, show_plot=False)
    # ave_interection = 0.0
    # for roi_id in mc.keys():
    #     print(mc[roi_id]['intersection'][0])
    #     ave_interection += mc[roi_id]['intersection'][0]
    #
    # ave_interection = ave_interection / float(len(mc))
    # print(ave_interection)
    # locs = pd.read_csv(locs_path)
    # rois = roi_coords(locs_path, locs, 16.0)
    # roi = rois['0251-1224']
    # rois = random_rois(locs)

    #
    # # build the voronoi diagram
    # vor = build_voronoi(roi['locs'], show_plot=False)
    #
    # # segment the objects based on the result of monte carlo simulation
    # max_area = 3065
    # thresh = 1.0 / max_area
    # d = densityw(locs)
    # density_factor = thresh / d
    # min_size = 5
    # print("density_factor: {}".format(density_factor))
    # objects = voronoi_clustering(vor, density_factor, min_size)
    # opoints = objects['locs'].as_matrix(['x [nm]', 'y [nm]'])
    # ofig = plot_voronoi_diagram(objects['voronoi'], locs_df=objects['locs'])
    # # oax = ofig.axes[0]
    # # oax.plot(opoints[:, 0], opoints[:, 1], 'bo', alpha=.5)
    #
    # # retain only the objects that are 'clustered'
    # filtered = objects['locs'][objects['locs']['lk'] != -1]
    # print(filtered.index.max())
    # filt_points = filtered.as_matrix(['x [nm]', 'y [nm]'])
    # filt_values = filtered['density'].values
    #
    # # prepare the data structure for re-clustering
    # data = {'voronoi': vor['voronoi'], 'locs': filtered}
    #
    # # set parameters and find clusters in objects
    # density_factor = 20
    # min_size = 5
    # n_locs = len(roi['locs'].index)
    # clusters = voronoi_clustering(data, density_factor, min_size, num_locs=n_locs)
    # cpoints = clusters['locs'].as_matrix(['x [nm]', 'y [nm]'])
    # cfig = plot_voronoi_diagram(clusters['voronoi'], locs_df=clusters['locs'])
    # # cax = cfig.axes[0]
    # # cax.plot(cpoints[:, 0], cpoints[:, 1], 'bo', alpha=.5)
    #
    # plot_cluster_polygons(objects['locs'], figure=cfig)
    # plt.show()
