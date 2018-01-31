import numpy as np
from math import pi,floor
import scipy.optimize
from scipy.fftpack import fft2,ifft2,fftshift
from scipy.spatial.distance import cdist

# Least Squares fitting
def fit_function_LS(data, params, x, fn):
    result = params
    errorfunction = lambda p: fn(*p)(x) - data
    good = True
    [result, cov_x, infodict, mesg, success] = scipy.optimize.leastsq(errorfunction,params,full_output = 1, maxfev = 500)
    err = errorfunction(result)
    err = scipy.sum(err * err)
    if (success < 1) or (success > 4):
        print "Fitting problem!", success, mesg
        good = False
    return [result, cov_x, infodict, good]
    
def exponential(a,b,c):
    return lambda x: a + b*np.exp(-x/c)
    
def exponential_cosine(a,b,c,d):
    return lambda x: a + b*np.exp(-x/c)*np.cos((pi*x)/2/d)
    
def exponential_gaussian(a,b,c,d,e):
    rho = a*1e-6
    amp = 1/2/pi/b**2/rho
    return lambda x: amp*np.exp(-x**2/2/b**2) + c*np.exp(-x/d)+e
    
def fit_exponential(data, x, params):
    return fit_function_LS(data, params, x, exponential)    
    
def fit_exponential_cosine(data, x, params):
    return fit_function_LS(data, params, x, exponential_cosine)    

def fit_exponential_gaussian(data, x, params):
    return fit_function_LS(data, params, x, exponential_gaussian) 
    
def fit_correlation(data,x_range,solver,guess=None):
    result, cov_x, infodict, good = solver(data[1:],x_range[1:],guess)
    if solver.__name__ == 'fit_exponential':
        fit_func = exponential
    if solver.__name__ == 'fit_exponential_cosine':
        fit_func = exponential_cosine
    if solver.__name__ == 'fit_exponential_gaussian':   
        fit_func = exponential_gaussian
    curve = init_curve(result,x_range,fit_func)
    return curve,result
    
def init_curve(params,x_range, fit_func):
    curve = fit_func(*params)(x_range)
    return curve

def cart2pol(x,y,z):
    theta = np.arctan2(x,y)
    r = np.sqrt(x**2+y**2)
    v = z
    return theta,r,v

def pc_corr(image1=None,image2=None,region=None,rmax=100):
    """
    This is a generalised version of Matlab code from the
    publication:
    
    "Correlation Functions Quantify Super-Resolution Images 
    and Estimate Apparent Clustering Due to Over-Counting"
    Veatch et al, PlosONE, DOI: 10.1371/journal.pone.0031457
    
    This paper should be referenced in any publication that
    results from the use of this function

    """
    if (image1 is None) or (image2 is None):
        raise 'Input at least one image to calculate correlation', image1
    
    if region is None:
        raise 'must create an roi'
    else:
        # roi will be [xmin,xmax,ymin,ymax]
        mask = np.zeros(image1.shape,dtype='float64')
        mask[region[0]:region[1],region[2]:region[3]] = 1.0

    N1 = np.sum(image1*mask) # number of particles within mask
    N2 = np.sum(image2*mask) # number of particles within mask
    I1 = image1.astype('float64')       # convert to double
    I2 = image2.astype('float64')
    L1 = I1.shape[0]+rmax # size of fft2 (for zero padding)
    L2 = I1.shape[1]+rmax # size of fft2 (for zero padding)

    A = float(np.sum(mask))      # area of mask
    NP = np.real(fftshift(ifft2(np.absolute(fft2(mask, (L1, L2)))**2)))# Normalization for correct boundary conditions
    with np.errstate(divide='ignore',invalid='ignore'):
        G1 = A**2/N1/N2*np.real(fftshift(ifft2(fft2(I1*mask,(L1, L2))*np.conj(fft2(I2*mask, (L1, L2))))))/NP
        
    G1[np.isnan(G1)] = 0.0
    xmin = floor(float(L1)/2)-rmax
    ymin = floor(float(L2)/2)-rmax
    w = h = 2*rmax+1
    G = G1[ymin:ymin+h,xmin:xmin+w]  #only return valid part of G
    xvals = np.ones((1, 2*rmax+1)).T*np.linspace(-rmax,rmax,2*rmax+1)    #map to x positions with center x=0
    yvals = np.outer(np.linspace(-rmax,rmax,2*rmax+1),np.ones((1, 2*rmax+1)))    #map to y positions with center y=0
    zvals = G
    theta,r,v = cart2pol(xvals,yvals,zvals)  # convert x, y to polar coordinates
    Ar = np.reshape(r,(1, (2*rmax+1)**2))
    Avals = np.reshape(v,(1, (2*rmax+1)**2))
    ind = np.argsort(Ar,axis=1)
    rr = np.sort(Ar,axis=1)
    vv = Avals[:,ind[0,:]]
    rbins = [i for i in range(int(floor(np.max(rr))))]      # the radii you want to extract
    bin = np.digitize(rr[0,:], rbins, right=True)      # bin by radius
    g = np.ones((1,rmax+1))
    dg = np.ones((1,rmax+1)) 
    for j in range(0,rmax+1):
        m = bin == j
        n2 = np.sum(m)             # the number of pixels in that bin
        if n2==0:
            continue
        else:
            g[:,j] = np.sum(m*vv)/n2   # the average G values in this bin
            dg[:,j] = np.sqrt(np.sum(m*(vv-g[:,j])**2))/n2 # the variance of the mean
            
    r = np.arange(rmax+1)
    G[rmax+1, rmax+1] = 0
    return g,r

def create_distance_matrix(pointsA,pointsB):   
    return np.sort(cdist(np.array(pointsA),np.array(pointsB),'euclidean'))

def ripleyLfunction(dataXY,dist_scale,box=None,method=0):
    
    if box is None:
        xmin = np.min(dataXY[:,0])
        ymin = np.min(dataXY[:,1])
        xmax = np.max(dataXY[:,0])
        ymax = np.max(dataXY[:,1])
        box = [xmin, xmax, ymin, ymax]
        
    N,cols = dataXY.shape  
    a = dataXY[:,0].T - box[0]
    a.shape = (1,a.shape[0])
    b = box[1] - dataXY[:,0].T
    b.shape = (1,b.shape[0])
    c = dataXY[:,1].T - box[2]
    c.shape = (1,c.shape[0])
    d = box[3] - dataXY[:,1].T
    d.shape = (1,d.shape[0])
    rbox = np.amin((np.concatenate((a,b,c,d),axis=0)),axis=0)
    A = (box[1]-box[0])*(box[3]-box[2])
    dist = create_distance_matrix(dataXY,dataXY).T
    if method == 0: # no correction...
        L = np.zeros((dist_scale.shape[0],1))
        for i in range(dist_scale.shape[0]):
            K = A*np.sum(np.sum(dist[1:,:]<dist_scale[i],axis=0))/N**2
            L[i] = np.sqrt(K/pi) - dist_scale[i]  
    elif method == 1: # edge correction
        L = np.zeros((dist_scale.shape[0],1))
        Nk = dist_scale.shape[0]
        for i in range(Nk):
            index = np.where(rbox > dist_scale[i])[0]
            if index.any():
                K = A*np.sum(np.sum(dist[1:,index]<dist_scale[i],axis=0))/(index.shape[0]*N)
                L[i] = np.sqrt(K/pi) - dist_scale[i]
    return L

def ripley_ci(num_locs,box,num_simulations,rmax,method):
    dist_scale = np.linspace(0, rmax, 200)
    lrand = np.zeros((dist_scale.shape[0],num_simulations))
    for s in range(num_simulations):
        rand_datax = np.random.uniform(box[0],box[1],num_locs)
        rand_datay = np.random.uniform(box[2],box[3],num_locs)
        rand_xy = np.zeros((num_locs,2))
        rand_xy[:,0] = rand_datax
        rand_xy[:,1] = rand_datay
        lrand[:,s] = ripleyLfunction(rand_xy,dist_scale,box,method)[:,0]
        
    meanl = np.mean(lrand, axis=1)
    stdl = np.std(lrand, axis=1)
    ci_plus = meanl + 2*stdl
    ci_minus = meanl - 2*stdl
    
    return ci_plus,ci_minus

def ripleyLperpoint(dataXY,xK,box,method):

    N,cols = dataXY.shape
    a = dataXY[:,0].T - box[0]
    a.shape = (1,a.shape[0])
    b = box[1] - dataXY[:,0].T
    b.shape = (1,b.shape[0])
    c = dataXY[:,1].T - box[2]
    c.shape = (1,c.shape[0])
    d = box[3] - dataXY[:,1].T
    d.shape = (1,d.shape[0])
    rbox = np.amin((np.concatenate((a,b,c,d),axis=0)),axis=0)
    A = (box[1]-box[0])*(box[3]-box[2])
    dist = create_distance_matrix(dataXY,dataXY).T

    if method == 0: # no correction...
        Lpp = np.zeros(N)
        for i in range(N):
            K = A*np.sum(dist[1:,i]<xK,axis=0)/float(N)
            Lpp[i] = np.sqrt(K/pi) - xK  
    elif method == 1: # edge correction
        Lpp = np.zeros((N,1))
        for k in range(N):
            I = np.where(rbox > xK)[0]
            if I.any():
                K = A*np.sum(dist[1:,I]<xK,axis=0)/float(I.shape[0])
                Lpp[k] = np.sqrt(K/pi) - xK

    return Lpp