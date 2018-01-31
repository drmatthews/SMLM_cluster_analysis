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
import math
import numpy
import pandas as pd
from matplotlib import pylab as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPoint, MultiLineString
from shapely.ops import cascaded_union, polygonize
from descartes import PolygonPatch


def voronoi(localisations, density_factor, min_size, show_plot=True, verbose=True):

    # this only works on the whole dataset at present - not any defined
    # ROIs. This may cause a problem when the number of localisations
    # exceeds 1e6 (will slow down dramatically)

    locs_df = localisations
    n_locs = locs_df.shape[0]
    points = locs_df.as_matrix(columns=['x [nm]', 'y [nm]'])
    x_range = numpy.max(points[:, 0]) - numpy.min(points[:, 0])
    y_range = numpy.max(points[:, 1]) - numpy.min(points[:, 1])
    a = x_range * y_range
    print("n_locs", n_locs)
    print("image area", a)
    img_density = float(n_locs) / a
    

    print("Creating Voronoi object.")
    vor = Voronoi(points)

    print("Calculating 2D region sizes.")
    for i, region_index in enumerate(vor.point_region):
        if ((i%10000) == 0):
            print("Processing point", i)

        vertices = []
        for vertex in vor.regions[region_index]:
        
            # I think these are edge regions?
            if (vertex == -1):
                vertices = []
                break

            vertices.append(vor.vertices[vertex])
            
        if (len(vertices) > 0):
            area = Polygon(vertices).area
            locs_df.set_value(i, 'area', area)

    # Used median density based threshold.
    ave_density = img_density #locs_df['a'].median()

    if verbose:
        print("Min density", 1.0 / locs_df['area'].min())
        print("Max density", 1.0 / locs_df['area'].max())
        print("Median density", 1.0 / locs_df['area'].median())
        print("ave image density", img_density)

    # Record the neighbors of each point.
    max_neighbors = 40
    neighbors = numpy.zeros((n_locs, max_neighbors), dtype = numpy.int32) - 1
    neighbors_counts = numpy.zeros((n_locs), dtype = numpy.int32)

    print("Calculating neighbors")
    for ridge_p in vor.ridge_points:

        p1 = ridge_p[0]
        p2 = ridge_p[1]

        # Add p2 to the list for p1
        neighbors[p1,neighbors_counts[p1]] = p2
        neighbors_counts[p1] += 1

        # Add p1 to the list for p2
        neighbors[p2,neighbors_counts[p2]] = p1
        neighbors_counts[p2] += 1

    if False:
        n1 = neighbors[0,:]
        print(n1)
        print(neighbors[n1[0],:])
    
    # use neighbours to calculate first rank density
    density = numpy.divide(
        numpy.add(1, neighbors_counts).astype(float), locs_df['area'].values
    )
    locs_df = locs_df.assign(density=pd.Series(density))

    # Mark connected points that meet the minimum density criteria.
    print("Marking connected regions")
    locs_df = locs_df.assign(lk=pd.Series(numpy.array([-1 for i in range(0,n_locs)])).values)
    min_density = density_factor * ave_density
    visited = numpy.zeros((n_locs), dtype = numpy.int32)

    def neighborsList(index):
        nlist = []
        for i in range(neighbors_counts[index]):
            loc_index = neighbors[index,i]
            if (visited[loc_index] == 0):
                nlist.append(neighbors[index,i])
                visited[loc_index] = 1
        return nlist

    cluster_id = 1
    for i in range(n_locs):
        if (visited[i] == 0):
            if (locs_df.iloc[i]['density'] > min_density):
                cluster_elt = [i]
                c_size = 1
                visited[i] = 1
                to_check = neighborsList(i)
                while (len(to_check) > 0):

                    # Remove last localization from the list.
                    loc_index = to_check[-1]
                    to_check = to_check[:-1]

                    # If the localization has sufficient density add to cluster and check neighbors.
                    if (locs_df.iloc[loc_index]['density'] > min_density):
                        to_check += neighborsList(loc_index)
                        cluster_elt.append(loc_index)
                        c_size += 1

                    # Mark as visited.
                    visited[loc_index] = 1

                # Mark the cluster if there are enough localizations in the cluster.
                if (c_size > min_size):
                    # print("cluster", cluster_id, "size", c_size)
                    for elt in cluster_elt:
                        locs_df.set_value(elt, 'lk', cluster_id)
                    cluster_id += 1
            visited[i] = 1
            
    if show_plot:
        plot_voronoi_diagram(vor, locs_df)

    print(cluster_id - 1, "clusters")

    return locs_df

def alpha_shape(points, alpha):
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

        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

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
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def plot_polygon(polygon, figure=None):
    if figure is None:
        from matplotlib import pylab as plt
        fig = plt.figure(figsize=(10,10))
    else:
        fig = figure
    ax = fig.add_subplot(111)
    margin = .3
    if polygon.bounds:
        x_min, y_min, x_max, y_max = polygon.bounds
        ax.set_xlim([x_min-margin, x_max+margin])
        ax.set_ylim([y_min-margin, y_max+margin])
        patch = PolygonPatch(polygon, fc='#999999',
                             ec='#000000', fill=True,
                             zorder=-1)
        ax.add_patch(patch)
    return fig


def plot_voronoi_diagram(vor, locs_df):
    v2d = voronoi_plot_2d(vor, show_points=False, show_vertices=False)
    cluster_locs_df = locs_df[locs_df['lk'] != -1]
    labels = cluster_locs_df['lk'].unique()
    ax = v2d.axes[0]
    for m in labels:
        cluster_points = cluster_locs_df[cluster_locs_df['lk'] == m].as_matrix(columns=['x [nm]', 'y [nm]'])

        concave_hull, edge_points = alpha_shape(cluster_points,
                                            alpha=0.01)

        patch = PolygonPatch(concave_hull, fc='#999999',
                             ec='#000000', fill=True,
                             zorder=-1)
        ax.add_patch(patch)
        ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'rx')
    #     lines = LineCollection(edge_points,color=mcolors.to_rgba('b'),linestyle='solid')
    #     ax.add_collection(lines)   

    plt.show()    