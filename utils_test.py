import os
import pandas as pd
from matplotlib import pyplot as plt
from cluster import utils
from read_roi import read_roi_file
from read_roi import read_roi_zip

root_dir = 'C:\\Users\\NIC ADMIN\\Documents\\dSTORM_third_run\\'
path = root_dir + 'processed\\run03_control_voronoi.xlsx'
obj_locs = utils.import_voronoi_clusters(path, sheetname='image')
clust_locs = utils.import_voronoi_clusters(path, sheetname='image', column='cluster_id')
obj_xy = obj_locs.as_matrix(['x [nm]', 'y [nm]'])
clust_xy = clust_locs.as_matrix(['x [nm]', 'y [nm]'])
fig = plt.figure()
plt.plot(obj_xy[:, 0], obj_xy[:, 1], 'ro', alpha=.5)
# plt.plot(clust_xy[:, 0], clust_xy[:, 1], 'bo', alpha=.5)
utils.plot_cluster_polygons(obj_locs, figure=fig, cluster_column='object_id')
utils.plot_cluster_polygons(clust_locs, figure=fig, cluster_column='cluster_id', patch_colour='#5ff442')
plt.show()
# print(cluster_list[0].x)

# roi_path = 'C:\\Users\\NIC ADMIN\\Documents\\atto647n storm data 180111\\filtered\\hawk run09 wtTNF_filtered_roiset.zip'
# pixel_size = 31.25
# utils.plot_clusters_in_roi(roi_path, pixel_size, cluster_list)
