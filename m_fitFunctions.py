# Licence: CC-BY-NC
# Author: Carola Fischer

import numpy as np

def get_functions(run_4D, fitorder_msac, fitorder_corr):
# Creates function handles for polynomial fits up to 3rd order
# for PC-MRI 2D or 4D flow data background phase correction using MSAC
# 4D flow:  each velocity encoding dimension is fitted individually
#
# Input:
# -----
# run_4D         - boolean to distinguish between 2D or 4D flow
#
# fitorder_msac  - int 0-3; polymial fit order used during MSAC
#
# fitorder_corr  - int 0-3; polymial fit order used for background phase
#                  correction
#
# Output:
# ------
# mfunc         - function handle, fit MSAC
#
# cfunc         - function handle, fit correction
#
# dfunc         - function handle, evalFunc MSAC (computes residuals)
#
# fmfunc        - function handle, use fitted MSAC model on data
#
# fcfunc        - function handle, use fitted correction model on data
#
    if run_4D:
        #fit MSAC
        mfunc = lambda points, **kwargs: fit4D(fitorder_msac, points, **kwargs)
        #fit correction
        cfunc = lambda points, **kwargs: fit4D(fitorder_corr, points, **kwargs)

        #feval MSAC
        fmfunc = lambda coeffs, points: eval4D(fitorder_msac, coeffs, points)
        #feval correction
        fcfunc = lambda coeffs, points: eval4D(fitorder_corr, coeffs, points)
        #distance function MSAC
        dfunc = lambda coeffs, points: dist4D(fitorder_msac, coeffs, points)
    else:
        #fit MSAC
        mfunc = lambda points, **kwargs: fit2D(fitorder_msac, points, **kwargs)
        #fit correction
        cfunc = lambda points, **kwargs: fit2D(fitorder_corr, points, **kwargs)

        #feval MSAC
        fmfunc = lambda coeffs, points: eval2D(fitorder_msac, coeffs, points)
        #feval correction
        fcfunc = lambda coeffs, points: eval2D(fitorder_corr, coeffs, points)
        #distance function MSAC
        dfunc = lambda coeffs, points: dist2D(fitorder_msac, coeffs, points)
    return mfunc, cfunc, fmfunc, fcfunc, dfunc


def getInOut4D(order, points):
    #separate input / output
    xyz = points[:,0:3]
    p1 = points[:,3]
    p2 = points[:,4]
    p3 = points[:,5]
    noP = points.shape[0]
    #get minimal samplesSize and input configs for different fit orders
    if order == 0:
        A = np.ones([noP, 1])
        no = 1
    elif order == 1:
        A = np.ones([noP, 4])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p3
        no = 4
    elif order == 2:
        A = np.ones([noP, 10])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p3
        A[:,4] = p1**2
        A[:,5] = p1*p2
        A[:,6]=p1*p3
        A[:,7] = p2**2
        A[:,8] = p2*p3
        A[:,9]=p3**2
        no = 10
    elif order == 3:
        A = np.ones([noP, 20])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p3
        A[:,4] = (p1**2)
        A[:,5] = p1*p2
        A[:,6] = p1*p3
        A[:,7] = (p2**2)
        A[:,8] = p2*p3
        A[:,9] = (p3**2)
        A[:,10] = p1**3
        A[:,11] = (p1**2)*p2
        A[:,12] = (p1**2)*p3
        A[:,13] = p2**3
        A[:,14] = (p2**2)*p1
        A[:,15] = (p2**2)*p3
        A[:,16] = p3**3
        A[:,17] = (p3**2)*p1
        A[:,18] = (p3**2)*p2
        A[:,19] = p1*p2*p3
        no = 20
    return xyz, A, no

def fit4D(order, points, **kwargs):
    inlierIndx = np.ones((points.shape[0], 3), dtype=bool) #select all points
    if len(kwargs) > 0: #define selection if not given
        inlierIndx = kwargs['inlierIndx']

    indx = inlierIndx == 1
    xyz, A, no = getInOut4D(order, points) #get input matrix A and output xyz
    coeffs = np.zeros((no, 3)) #define coefficent matrix
    for d in np.arange(3): #fit for each velocity dimension
        if order == 0:
            coeffs[0,d] = np.mean(xyz[indx[:,d],d], 0)
        else:
            inlier = indx[:,d]
            coeffs[:,d] = np.linalg.lstsq(A[inlier,:],xyz[inlier,d], rcond=None)[0]
    return coeffs


def dist4D(order, coeffs, points):
    est, xyz = eval4D(order, coeffs, points) #use a model on points
    dist = np.abs((xyz-est)/2) #get residuals
    return dist

def eval4D(order, coeffs, points):
    [xyz, A, no] = getInOut4D(order, points) #get input matrix A and output xyz
    est = A@coeffs #evaluate model on points in A
    return est, xyz

######### 2D #########
def getInOut2D(order, points):
    #separate input / output
    xyz = points[:,0:1]
    p1 = points[:,1]
    p2 = points[:,2]
    noP = points.shape[0]
    #get minimal samplesSize and input configs for different fit orders
    if order == 0:
        A = np.ones([noP, 1])
        no = 1
    elif order == 1:
        A = np.ones([noP, 3])
        A[:,1] = p1
        A[:,2] = p2
        no = 3
    elif order == 2:
        A = np.ones([noP, 6])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p1**2
        A[:,4] = p1*p2
        A[:,5] = p2**2
        no = 6
    elif order == 3:
        A = np.ones([noP, 10])
        A[:,1] = p1
        A[:,2] = p2
        A[:,3] = p1**2
        A[:,4] = p1*p2
        A[:,5] = p2**2
        A[:,6] = p1**3
        A[:,7] = (p1**2)*p2
        A[:,8] = p2**3
        A[:,9] = (p2**2)*p1
        no = 10
    return xyz, A, no

def fit2D(order, points, **kwargs):

    inlierIndx = np.ones((points.shape[0],1), dtype=bool) #select all points
    if len(kwargs) > 0: #define selection if not given
        inlierIndx = kwargs['inlierIndx']

    indx = (inlierIndx == 1)[:,0]
    xyz, A, no = getInOut2D(order, points) #get input matrix A and output xyz
    coeffs = np.zeros((no,1)) #define coefficent matrix

    if order == 0:
        coeffs[0,0] = np.mean(xyz[indx,0], 0)
    else:
        coeffs[:,0]= np.linalg.lstsq(A[indx,:],xyz[indx,0], rcond=None)[0]
    return coeffs


def dist2D(order, coeffs, points):
    est, xyz = eval2D(order, coeffs, points) #use a model on points
    dist = np.abs((xyz-est)/2) #get residuals
    return dist

def eval2D(order, coeffs, points):
    [xyz, A, no] = getInOut2D(order, points) #get input matrix A and output xyz
    est= A@coeffs #evaluate model on points in A
    return est, xyz
