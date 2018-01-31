import os
from cluster import utils
from read_roi import read_roi_file
from read_roi import read_roi_zip

root_dir = 'C:\\Users\\NIC ADMIN\\Documents\\atto647n storm data 180111\\'
path = root_dir + 'processed\\hawk run09 wtTNF_filtered_optics.xlsx'
cluster_list = utils.import_clusters(path)
# print(cluster_list[0].x)

roi_path = 'C:\\Users\\NIC ADMIN\\Documents\\atto647n storm data 180111\\filtered\\hawk run09 wtTNF_filtered_roiset.zip'
pixel_size = 31.25
utils.plot_clusters_in_roi(roi_path, pixel_size, cluster_list)