from __future__ import absolute_import, print_function

import os
import math
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import scipy.special as special
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay
from .optics_ import OPTICS
from .triangulation import pc_ratio
from shapely.geometry import MultiPoint, MultiLineString
from shapely.ops import cascaded_union, polygonize
from matplotlib import pylab as plt
# import the imagej roi
from read_roi import read_roi_file
from read_roi import read_roi_zip
import os


class Cluster(object):
    def __init__(self, coordinates, cluster_id):
        N = coordinates.shape[0]
        self.x = np.reshape(coordinates[:, 0], (N, 1))
        self.y = np.reshape(coordinates[:, 1], (N, 1))
        self.cluster_id = cluster_id
        id_array = np.array(
            [cluster_id for i in range(coordinates.shape[0])],
            dtype='int32')
        self.id_array = np.reshape(id_array, (N, 1))
        self.occupancy = coordinates.shape[0]
        self.alpha = 0.01
        self.hull, edges = self._alpha_shape(coordinates, self.alpha)
        self.pc_ratio = None
        self.area = None
        self.perimeter = None
        self.nn = None
        self.center = None
        self.is_valid = False
        if edges:
            self.edges = np.vstack([np.reshape(a, (1, 4)) for a in edges])
            self.pc_ratio = self._pc(edges)
            self.area = self.hull.area
            self.perimeter = self.hull.length
            self.nn = np.reshape(self._near_neighbours(coordinates), (N, 1))
            self.center = self._center(np.hstack((self.x, self.y)))
            self.is_valid = True
        else:
            self.edges = edges

    def _alpha_shape(self, points, alpha):
        """
        Compute the alpha shape (concave hull) of a set
        of points.
        @param points: Numpy array of object coordinates.
        @param alpha: alpha value to influence the
            gooeyness of the border. Smaller numbers
            don't fall inward as much as larger numbers.
            Too large, and you lose everything!
        """
        if len(points) < 4:
            # When you have a triangle, there is no sense
            # in computing an alpha shape.
            return MultiPoint(list(points)).convex_hull

        def add_edge(edges, edge_points, coords, i, j):
            """
            Add a line between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                return

            edges.add((i, j))
            edge_points.append(coords[[i, j]])

        # coords = numpy.array([point.coords[0] for point in points])
        coords = points
        tri = Delaunay(coords)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the
        # triangle
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]

            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
            b = math.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
            c = math.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
            # Semiperimeter of triangle
            s = (a + b + c) / 2.0
            # Area of triangle by Heron's formula
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))

            if area > 0.0:
                circum_r = a * b * c / (4.0 * area)
                # Here's the radius filter.
                # print circum_r
                if circum_r < 1.0 / alpha:
                    add_edge(edges, edge_points, coords, ia, ib)
                    add_edge(edges, edge_points, coords, ib, ic)
                    add_edge(edges, edge_points, coords, ic, ia)
            else:
                continue
        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        return cascaded_union(triangles), edge_points

    def _near_neighbours(self, points):
        tree = KDTree(points, leaf_size=2)
        dist, ind = tree.query(points[:], k=2)
        return dist[:, 1]

    def _pc(self, edges):
        e = np.vstack(edges)
        Y = np.c_[e[:, 0], e[:, 1]]
        pcr = pc_ratio(Y)
        return pcr

    def _center(self, xy):
        kmeans = KMeans(n_clusters=1, random_state=0).fit(xy)
        return kmeans.cluster_centers_[0]


class ClusterList(object):
    def __init__(self):
        self.clusters = []
        self.n_clusters = len(self.clusters)
        self.noise = None

    def __getitem__(self, i):
        return self.clusters[i]

    def __setitem__(self, item):
        if isinstance(item, Cluster):
            self.clusters.append(item)
            self.n_clusters = len(self.clusters)

    def __len__(self):
        return self.n_clusters

    def append(self, item):
        self.clusters.append(item)
        self.n_clusters = len(self.clusters)

    def save(self, path, sheetname='image'):

        ext = os.path.splitext(path)[1]
        if ('xlsx' in ext):
            if self.clusters:
                
                writer = pd.ExcelWriter(path, engine='openpyxl')
                if os.path.exists(path):
                    book = load_workbook(path)
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

                coords = []
                stats = []
                hulls = []
                for c in self.clusters:
                    coords.append(np.hstack([c.x, c.y, c.nn, c.id_array]))
                    stats.append(np.hstack(
                        [c.area, c.perimeter, c.pc_ratio, c.occupancy]))
                    hull_cluster_id = np.array(
                        [c.cluster_id
                         for i in range(c.edges.shape[0])], dtype='int32')
                    hull_cluster_id = np.reshape(
                        hull_cluster_id, (c.edges.shape[0], 1))
                    hulls.append(np.hstack([c.edges, hull_cluster_id]))

                c_df = pd.DataFrame(np.vstack(coords))
                c_df.columns = [
                    'x [nm]', 'y [nm]', 'nn distance [nm]', 'cluster_id']
                s_df = pd.DataFrame(np.vstack(stats))
                s_df.columns = [
                    'area [nm^2]', 'perimeter [nm]', 'pc_ratio', 'occupancy']
                h_df = pd.DataFrame(np.vstack(hulls))
                h_df.columns = [
                    'x0 [nm]', 'y0 [nm]', 'x1 [nm]', 'y1 [nm]', 'cluster_id']

                c_df.to_excel(
                    writer,
                    sheet_name='{} cluster coordinates'.format(sheetname),
                    index=False
                )
                s_df.to_excel(
                    writer,
                    sheet_name='{} cluster statistics'.format(sheetname),
                    index=False
                )
                h_df.to_excel(
                    writer,
                    sheet_name='{} convex hulls'.format(sheetname),
                    index=False
                )
                
                if self.noise is not None:
                    noise_df = pd.DataFrame(self.noise)
                    noise_df.columns = ['x [nm]', 'y [nm]']
                    noise_df.to_excel(
                        writer,
                        sheet_name='{} noise'.format(sheetname),
                        index=False
                    )
                writer.save()
            else:
                print("no clusters to save")
        else:
            print("file path for saving must contain extension xlsx")


def optics_clustering(input_coords,
                      eps=None,
                      eps_extract=None,
                      min_samples=1,
                      metric="euclidean"):

    optics_clusters = {}
    noise = {}
    for roi in input_coords.keys():
        xy = input_coords[roi]
        
        if xy.shape[0] == 0:
            print("no coords in roi")
            continue

        if eps is None:
            eps = _epsilon(xy, min_samples)

        clusters = OPTICS(eps=eps, min_samples=min_samples, metric=metric)
        clusters.fit(xy)

        if eps_extract:
            clusters.extract(eps_extract, 'dbscan')

        core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
        core_samples_mask[clusters.core_sample_indices_] = True
        unique_labels = set(clusters.labels_)

        cluster_list = ClusterList()
        for m in unique_labels:

            class_member_mask = (clusters.labels_ == m)
            if m != -1:
                coords = xy[class_member_mask & core_samples_mask]
                if coords.shape[0] > 3:
                    cluster_ = Cluster(coords, m)
                    if cluster_.is_valid:
                        cluster_list.append(Cluster(coords, m))
                        optics_clusters[roi] = cluster_list
            else:
                noise_ = xy[class_member_mask & ~core_samples_mask]
                cluster_list.noise = noise_
                noise[roi] = noise_
            
    return (optics_clusters, noise)


def _epsilon(x, k):
    if len(x.shape) > 1:
        m, n = x.shape
    else:
        m = x.shape[0]
        n == 1
    prod = np.prod(x.max(axis=0) - x.min(axis=0))
    gamma = special.gamma(0.5 * n + 1)
    denom = (m * np.sqrt(np.pi**n))
    Eps = ((prod * k * gamma) / denom)**(1.0 / n)

    return Eps


# helpers
def kdtree_nn(points):
    tree = KDTree(points, leaf_size=2)
    dist, ind = tree.query(points[:], k=2)
    return dist[:, 1]


def plot_clusters_in_roi(roi_path,
                         pixel_size,
                         cluster_list,
                         noise=None,
                         save=False,
                         filename=None,
                         show_plot=True,
                         new_figure=True,
                         cluster_marker_size=4):

    filename, ext = os.path.splitext(roi_path)
    print(ext)
    if 'zip' in ext:
        rois = read_roi_zip(roi_path)
    else:
        rois = read_roi_file(roi_path)

    for roi_id, roi in rois.items():
        for k, v in roi.items():
            if not isinstance(v, str):
                roi[k] = float(v) * pixel_size

        plot_optics_clusters(
                cluster_list,
                noise=noise,
                save=False,
                filename=None,
                show_plot=True,
                new_figure=True,
                cluster_marker_size=4,
                roi=roi)


def plot_optics_clusters(cluster_list,
                         noise=None,
                         save=False,
                         filename=None,
                         show_plot=True,
                         new_figure=True,
                         cluster_marker_size=4,
                         roi=None):

    if new_figure:
        plt.figure()

    colors = plt.cm.Spectral(np.linspace(0, 1, cluster_list.n_clusters))
    num_clusters = cluster_list.n_clusters
    if roi is not None:
        num_clusters = 0

    for c, color in zip(cluster_list, colors):
        if roi is not None:
            # plot clusters and annotate centroids
            coords = np.hstack((c.x, c.y))
            coords = coords[(coords[:, 0] > roi['left']) &
                            (coords[:, 0] < roi['left'] + roi['width']) &
                            (coords[:, 1] > roi['top']) &
                            (coords[:, 1] < roi['top'] + roi['height'])]
            if coords.shape[0] > 0:
                num_clusters += 1
                x = coords[:, 0]
                y = coords[:, 1]
            else:
                continue
        else:
            x = c.x
            y = c.y

        plt.plot(x, y, 'o',
            markerfacecolor=color,
            markeredgecolor='k',
            markersize=cluster_marker_size,
            alpha=0.5)

        plt.annotate('%s' % str(c.cluster_id),
            xy=(c.center[0:2]),
            xycoords='data')

    # plot noise
    if noise is not None:
        plt.plot(noise[:, 0], noise[:, 1], 'o', markerfacecolor='k',
                 markeredgecolor='k', markersize=1, alpha=0.5)

    plt.title('Estimated number of clusters: %d' % num_clusters)
    plt.xlabel('X [nm]')
    plt.ylabel('Y [nm]')
    plt.gca().invert_yaxis()

    if save and filename is not None:
        plt.savefig(filename)

    if show_plot:
        plt.show()


def plot_voronoi_clusters(v, vor, save=False, filename=None, show_plot=True):
    voronoi_plot_2d(vor, show_points=True, show_vertices=False)
    cluster_locs_df = v[v['lk'] != -1]
    labels = cluster_locs_df['lk'].unique()

    for m in labels:
        cluster_points = cluster_locs_df[
            cluster_locs_df['lk'] == m].as_matrix(columns=['x', 'y'])
        hull = ConvexHull(cluster_points)
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'ko')
        for simplex in hull.simplices:
            plt.plot(
                cluster_points[simplex, 0], cluster_points[simplex, 1], 'r-')

    if save and filename is not None:
        plt.savefig(filename)

    if show_plot:
        plt.show()

def import_clusters(path, sheetname=None):
    if path.endswith('xlsx'):
        cluster_sn = 'cluster coordinates'
        cluster_sn = 'noise'
        if sheetname:
            cluster_sn = '{0} cluster coordinates'.format(sheetname)
            noise_sn = '{0} noise'.format(sheetname)  
        clusters_df = pd.read_excel(path, sheet_name=cluster_sn)
        noise_df = pd.read_excel(path, sheet_name=noise_sn)
        clusters = df.groupby('cluster_id')
        cluster_list = ClusterList()
        for cid, cluster in clusters:
            coords = cluster.as_matrix(['x [nm]', 'y [nm]'])
            cluster_list.append(Cluster(coords, cid))
        cluster_list.noise = noise_df.as_matrix(['x [nm]', 'y [nm]'])
        return cluster_list
    else:
        raise ValueError("Input data must be in Excel format")

if __name__ == '__main__':
	root_dir = 'C:\\Users\\NIC ADMIN\\Documents\\atto647n storm data 180111\\'
	path = root_dir + 'processed\\hawk run09 wtTNF_filtered_optics.xlsx'
	cluster_list = import_clusters(path)

    # roi_path = '/home/daniel/Documents/Image Processing/Penny/dSTORM/atto647n storm data/RoiSet.zip'
    # pixel_size = 31.25  # nm

    # plot_clusters_in_roi(roi_path,
                         # pixel_size,
                         # noise=None,
                         # save=False,
                         # filename=None,
                         # show_plot=True,
                         # new_figure=True,
                         # cluster_marker_size=4)

    # roi_path = '/home/daniel/Documents/Image Processing/Penny/dSTORM/atto647n storm data/hawk.roi'
    # pixel_size = 31.25  # nm

    # plot_clusters_in_roi(roi_path,
                         # pixel_size,
                         # noise=None,
                         # save=False,
                         # filename=None,
                         # show_plot=True,
                         # new_figure=True,
                         # cluster_marker_size=4)