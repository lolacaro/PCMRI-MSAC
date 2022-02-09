# Licence: CC-BY-NC
# Author: Carola Fischer
#imports
import numpy as np
import time
from m_fitFunctions import get_functions
from m_msac import msac

def run(phase_im_t, magnitude_im_t, parameters):
# Preps 2D or 4D flow data, runs MSAC and corrects velocity data
# with fit from MSAC derived stationary tissue mask
#
# Input:
# -----
# phase_im_t      - double array, time-resolved phase data 2D or 4D flow
#                   normed to [-1,1[
#                   dimensions: [dim1 dim2 slice time velEnc]
#
# magnitude_im_t  - double array, time-resolved magnitude data 2D or
#                   4D flow normed to [0,1]
#                   dimensions: [dim1 dim2 slice time velEnc]
#
# parameters      - dict containing necessary Parameters for MSAC
#                   mgn_thresh:     double, for thresholding magnitude image
#                   msac_thresh:    double, MSAC threshold parameter
#                   samples:        # of samples points drawn in MSAC
#                   trials:         # of trials in MSAC
#                   msac_fit_order: polynomial fit order in MSAC
#                   corr_fit_order: polynomial fit order for correction
#
# Output:
# ------
# results      - dict with the following fields
#                time_total: time for complete function
#                time_msac:  time for msac
#                mask_mgn:   magnitude mask
#                mask_msac:  stationary tissue derived by MSAC
#                model:      correction fit model using msac mask
#                bgr_msac:   background fit msac
#                corr_avg:   corrected phase data averaged
#                corr_tres:  corrected phase data time-resolved
#


    print('RUN: Prep MSAC ...')
    time1 = time.time()
    results = dict()
    run_4D = parameters['flow_dimensions'] == 3

    m,n,l,t = magnitude_im_t.shape #dimensions
    # time-average data
    magnitude = np.squeeze(np.mean(magnitude_im_t,3))
    phase = np.squeeze(np.mean(phase_im_t,3))

    #create magnitude mask
    results['mask_mgn'] = magnitude > parameters['mgn_thresh']

    #create [dat x y z] point-arrays for all pixels ap
    #and pixels in magnitude mask mp
    #needed to run MSAC
    d = parameters['flow_dimensions']
    if run_4D:
        allm = np.ones([m,n,l])
        aa = np.argwhere(allm == 1) #all data points
        ma = np.argwhere(results['mask_mgn'] == 1) #magnitude points

        ap = np.zeros((aa.shape[0], 6))
        mp = np.zeros((ma.shape[0], 6))
        ap[:,3:6] = aa
        mp[:,3:6] = ma
        for dir in np.arange(d):
            pdir = np.squeeze(phase[:,:,:,dir])
            ap[:,dir] = pdir[allm == 1]
            mp[:,dir] = pdir[results['mask_mgn'] == 1]

    else:
        allm = np.ones([m,n])
        aa = np.argwhere(allm == 1) #all data points
        ma = np.argwhere(results['mask_mgn'] == 1) #magnitude points

        ap = np.zeros((aa.shape[0], 3))
        mp = np.zeros((ma.shape[0], 3))
        ap[:,1:3] = aa
        mp[:,1:3] = ma

        ap[:,0] = phase[allm == 1]
        mp[:,0] = phase[results['mask_mgn'] == 1]



    #Define corr fit, msac fit, msac dist function and corr/msac feval functions
    mfunc, cfunc, fmfunc, fcfunc, dfunc = get_functions(run_4D, parameters['msac_fit_order'], parameters['corr_fit_order'])

    functions = dict()
    functions['msac_fit'] = mfunc
    functions['msac_dist'] = dfunc

    print('RUN: Start MSAC ...')
    time2 = time.time()

    #run MSAC // based on ransac and msac functions of MATLAB
    cost, inlierIndx = msac(mp, parameters, functions)
    results['time_msac'] = time.time() - time2

    #get correction fit
    print('RUN: Get and correct background phase ...')
    results['model'] = cfunc(mp, **{'inlierIndx': inlierIndx}) #fit using MSAC inliers
    est, xyz = fcfunc(results['model'], ap) #evaluate fit on image


    #define MSAC mask from inlierIndx returned by MSAC and background fit
    results['mask_msac'] = np.squeeze(np.zeros([m,n,l,d]))
    results['bgr_msac'] = np.squeeze(np.zeros([m,n,l,d]))
    aw = np.where(allm == 1) #all data points
    mw = np.where(results['mask_mgn'] == 1)
    #fill masks and fit-images
    if run_4D:
        results['mask_msac'][mw[0], mw[1], mw[2],:] = inlierIndx
        results['bgr_msac'][aw[0], aw[1], aw[2],:] = est
    else:
        results['mask_msac'][mw[0], mw[1]] = inlierIndx[:,0]
        results['bgr_msac'][aw[0], aw[1]] = est[:,0]


    #calculate corrected phase images
    results['corr_avg'] = phase - np.squeeze(results['bgr_msac'])

    if run_4D:
        corr_t = np.swapaxes(np.expand_dims(results['bgr_msac'], -1), 3, 4) #expand t, put in order m n l t d
    else:
        corr_t = np.expand_dims(results['bgr_msac'], axis = (2,3)) #expand l and t
    results['corr_tres'] = phase_im_t - corr_t
    results['inlierIndx'] = inlierIndx
    #stop time_total
    results['time_total'] = time.time() - time1

    return results
