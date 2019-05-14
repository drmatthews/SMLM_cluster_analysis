#from utilities import filetype_config
import os
import pandas as pd 
import numpy as np
from math import ceil

import ijroi

from cluster.voronoi_ import voronoi as voronoi

# import javabridge as jv
# import bioformats as bf
# from bioformats import log4j

# VM_STARTED = False
# VM_KILLED = False

# BF2NP_DTYPE = {
#     0: np.int8,
#     1: np.uint8,
#     2: np.int16,
#     3: np.uint16,
#     4: np.int32,
#     5: np.uint32,
#     6: np.float32,
#     7: np.double
# }

# def concatenate_df(df_list):
#     return pd.concat(df_list).reset_index(drop=True)

# def start_jvm(max_heap_size='4G'):

#     """
#     Start the Java Virtual Machine, enabling BioFormats IO.
#     Optional: Specify the path to the bioformats_package.jar to your needs by calling.
#     set_bfpath before staring to read the image data

#     Parameters
#     ----------
#     max_heap_size : string, optional
#     The maximum memory usage by the virtual machine. Valid strings
#     include '256M', '64k', and '2G'.
#     """
#     #log4j.basic_config()
#     jars = bf.JARS
#     jv.start_vm(class_path=jars, max_heap_size=max_heap_size)
#     VM_STARTED = True

# def kill_jvm():
#     """
#     Kill the JVM. Once killed, it cannot be restarted.
#     See the python-javabridge documentation for more information.
#     """
#     jv.kill_vm()
#     VM_KILLED = True

# def jvm_error():

#     raise RuntimeError("The Java Virtual Machine has already been "
#                        "killed, and cannot be restarted. See the "
#                        "python-javabridge documentation for more "
#                        "information. You must restart your program "
#                        "and try again.")

# class Reader:
#     # Close the file on cleanup.
#     def __del__(self):
#         if self.reader:
#             self.reader.close()

#     # can be used when employing "with"
#     def __enter__(self):
#         return self

#     def __exit__(self, etype, value, traceback):
#         if self.reader:
#             self.reader.close()    

# class NSTORMReader(Reader):
#     def __init__(self, path):
#         if not VM_STARTED:
#             start_jvm()
#         if VM_KILLED:
#             jvm_error()
#         print "initialising image reader and getting metadata. this can take a while - be patient."
#         self.filepath = path
#         self.filename = os.path.basename(path)
#         self.reader = bf.ImageReader(path)
#         self.metadata = None

#         self.sizeX = self.reader.rdr.getSizeX()
#         self.sizeY = self.reader.rdr.getSizeY()
#         self.sizeZ = self.reader.rdr.getSizeZ()
#         self.sizeT = self.reader.rdr.getSizeT()
#         self.sizeC = self.reader.rdr.getSizeC()        

#     def read(self, theT):
#         """
#         use bioformats reader.read() to get the
#         specified time point assuming there is
#         always just the one channel and one z plane
#         """     
#         if (self.sizeC == 1) and (self.sizeZ == 1):
#             return self.reader.read(c=0,z=0,t=theT,rescale=False)
#         else:
#             print "the dataset either has too many channels"
#             print "or too many z slices"
#             return None

#     def read_range(self, rangeT):
#         sizeT = len(rangeT)
#         shape = [self.sizeX, self.sizeY, sizeT]
#         planes = np.empty(shape, dtype=BF2NP_DTYPE[self.reader.getPixelType()])
#         i = 0
#         for theT in range(rangeT[0],rangeT[1]):
#             planes[:,:,i] = self.read(theT,rescale=False)

#         return planes

#     def get_ome_xml(self):
#         self.omexml = bf.get_omexml_metadata(self.filepath)
#         self.metadata = bf.OMEXML(self.omexml.encode('ascii','ignore'))
#         return self.metadata

#     def get_pixels(self):
#         if self.metadata:
#             self.pixels = self.metadata.image(0).Pixels
#             return self.pixels
#         else:
#             return None


# class Molecule(object):

#     def __init__(self):
#         self.x = None
#         self.y = None
#         self.t = None
#         self.amplitude = None
#         self.bg_mean = None
#         self.bg_Std = None
#         self.sigma = None
#         self.img = None
#         self.pixels = None
#         self.good = True
#         self.cov_x = None
#         self.info = None
#         self.resid = None
        
#     def set_pixels(self, image, window):
#         xstart = ceil(self.x - (window) / 2)
#         xstop = ceil(self.x + (window + 1) / 2)
#         ystart = ceil(self.y - (window) / 2)
#         ystop = ceil(self.y + (window + 1) / 2)
#         self.img = image[ystart:ystop, xstart:xstop]
#         self.pixels =  np.array([ystart, ystop, xstart, xstop])
        
#     def set_sigma(self,sigma):
#         self.sigma = sigma

def import_ij_rois(basename,roipath):

    rois = []
    for filename in os.listdir(roipath):
        if ('.roi' in filename) and (basename in filename):
            fpath = os.path.join(roipath,filename)
            print('roi path: ',fpath)
            with open(fpath, "rb") as f:
                r = ijroi.read_roi(f)
            # extract top left and bottom right of roi
            # reorganise to make a list of x0, y0, x1, y1
            roi = [r[0,1].astype('float'),r[0,0].astype('float'),\
                    r[2,1].astype('float'),r[2,0].astype('float')]
            rois.append(roi)
            localisations
    return rois

def import_rois(fpath):

    roidf = pd.read_csv(fpath)

    # and extract into a list
    rois = []
    for row in roidf.iterrows():
        i, d = row
        rois.append(d.tolist()[1:])
    
    return rois

def get_coords(localisations, roi, pixel_size):
    roinm = [r*pixel_size for r in roi]

    x_column = localisations.x_column
    y_column = localisations.y_column

    locs = localisations[(localisations[x_column] > roinm[0]) & (localisations[x_column] < roinm[2]) 
                   & (localisations[y_column] > roinm[1]) & (localisations[y_column] < roinm[3])] 
    return locs      

def get_localisations_in_roi(localisations, roi, pixel_size):
    if isinstance(rois[0],list):
        coord_list = []
        for roi in rois:
            roinm = [r*pixel_size for r in roi]
            xy = get_coords(localisations, roi, pixel_size)
            coord_list.append(xy)
        return coord_list
    else:
        return get_coords(localisations, rois, pixel_size)    
        
class Localisations(object):

    def __init__(self, filepath, source='nstorm', roi_params=None):
        try:
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.basename = os.path.splitext(self.filename)[0]
            self.source = source
            self.rois = None
            self._read_file(filepath)             
            self.num_localisations = len(self.data)
            if roi_params:
                self.pixel_size = roi_params['pixel_size']
                self.roi_path = roi_params['path']
                self.roi_source = roi_params['source']
                try:
                    if os.path.isdir(self.roi_path):
                        self.parse_rois(self.roi_path, self.roi_source, self.pixel_size)
                    else:
                        roi_ext = os.path.splitext(self.roi_path)[1]
                        if (self.basename in self.roi_path) and (self.roi_source in roi_ext):
                            if ('.roi' in roi_ext):
                                self.rois = self.parse_ij_roi(self.roi_path, self.pixel_size)
                            elif ('.csv' in roi_ext):
                                self.rois = self.parse_csv_roi(self.roi_path, self.pixel_size)
                except:
                    print("There was a problem parsing the roi(s). Check path and source.")
    
                self._mark_rois()
            self._get_points()
        except:
            print("There was a problem parsing the file")


    def _read_file(self, filepath):
        name,ext = os.path.splitext(filepath)
        if ('txt' in ext) and ('nstorm' in self.source):
            self.data = pd.read_table(filepath)
            self.x_column = 'Xc'
            self.y_column = 'Yc'
            self.uncertainty_column = 'Lateral Localization Accuracy'
            self.f_column = 'Frame'
        elif ('csv' in ext) and ('thunderstorm' in self.source):
            self.data = pd.read_table(filepath,delimiter=',')
            self.x_column = 'x [nm]'
            self.y_column = 'y [nm]'
            self.uncertainty_column = 'uncertainty [nm]'
            self.f_column = 'frame'
        if len(self.data) > 0:
            self.n_locs = self.data.shape[0]
        else:
            "Couldn't parse file - check source definition"

    def _get_points(self):
        columns = [self.x_column,self.y_column]
        if self.rois is not None:
            for r in range(self.rois.shape[0]):
                columns.append('roi_%s'%r)
        self.points = self.data.loc[:,columns]
        self.points.rename(columns={self.x_column: 'x', self.y_column: 'y'}, inplace=True)

    def _mark_rois(self):
        df = self.data
        x_column = self.x_column
        y_column = self.y_column
        for i, roi in enumerate(self.rois.iterrows()):
            df['roi_%s'%i] = (df[x_column] > roi[1]['x0 [pixels]']) & (df[x_column] < roi[1]['x1 [pixels]']) \
                       & (df[y_column] > roi[1]['y0 [pixels]']) & (df[y_column] < roi[1]['y1 [pixels]'])

    def parse_ij_roi(self, path, pixel_size):
        with open(path, "rb") as f:
            r = ijroi.read_roi(f)
        # extract top left and bottom right of roi
        # reorganise to make a list of x0, y0, x1, y1, id
        roi_df = pd.DataFrame([[r[0,1].astype('float'),r[0,0].astype('float'),\
                r[2,1].astype('float'),r[2,0].astype('float')]]) * pixel_size
        roi_df.columns = ['x0 [pixels]','y0 [pixels]','x1 [pixels]','y1 [pixels]']
        return roi_df

    def parse_csv_roi(self, path, pixel_size):
        roi_df = pd.read_csv(path,index_col=0)
        for col in roi_df.columns:
            roi_df[col] *= pixel_size
        roi_df.columns = ['x0 [pixels]','y0 [pixels]','x1 [pixels]','y1 [pixels]']
        return roi_df

    def parse_rois(self, roi_path, source, pixel_size):
        roi_df = pd.DataFrame()
        for f in os.listdir(roi_path):
            ext = os.path.splitext(f)[1]
            if (self.basename in f) and (source in ext):
                r = os.path.join(roi_path,f)
                if ('.roi' in ext):
                    roi_df = roi_df.append(self.parse_ij_roi(r, pixel_size))
                elif ('.csv' in ext):
                    roi_df = roi_df.append(self.parse_csv_roi(r, pixel_size))
        roi_df.columns = ['x0 [pixels]','y0 [pixels]','x1 [pixels]','y1 [pixels]']
        self.rois = roi_df

    def get_from_roi(self, roi_id=0, pixel_size=32):

        try:
            roi = self.rois.iloc[roi_id]

            df = self.data
            x_column = self.x_column
            y_column = self.y_column

            locs = df[(df[x_column] > roi['x0 [pixels]']) & (df[x_column] < roi['x1 [pixels]']) \
                           & (df[y_column] > roi['y0 [pixels]']) & (df[y_column] < roi['y1 [pixels]'])] 
            return locs        
        except:
            print("Couldn't find specified ROI")

    def get_header(self):
        return self.data.columns

if __name__ == '__main__':
    
    root_dir = '/home/daniel/Documents/Image Processing/Penny/PALM/'
    folder = 'test/'
    filename = 'starve 1_unfiltered_TS2D.csv'
    path = root_dir + folder + filename
    
    roi_params = {}
    roi_params['path'] = root_dir + folder + 'starve 1_unfiltered_TS2D.roi'
    roi_params['source'] = 'roi'
    roi_params['pixel_size'] = 32.0
    localisations = Localisations(path, source='thunderstorm', roi_params=roi_params)