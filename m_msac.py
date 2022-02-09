# Licence: CC-BY-NC
# Author: Carola Fischer
import numpy as np

def msac(points, parameters, functions):
# MSAC M-estimator SAmple Consensus (MSAC) algorithm for outlier
# sensitive stationary tissue masks for PC-MRI 2D or 4D flow data
# 4D flow: each velocity encoding dimension is fitted and updated individually
#
# Input:
# -----
# points      - M-by-3 or M-by-6 for [vel x y] or [vel1 vel2 vel3 x y z] data
#
# parameters  - dict containing the following fields:
#               samples
#               msac_thresh
#               trials
#               flow_dimensions
#
# functions   - dict containing the following function handles
#               msac_fit
#               msac_dist
#
# Output:
# ------
#
# bestInliers - logical array of length M, to mark inliers in points
#               derived by MSAC
#
# bestCost    - double, MSAC cost with above inliers
#
#
# References:
# ----------
#   P. H. S. Torr and A. Zisserman, "MLESAC: A New Robust Estimator with
#   Application to Estimating Image Geometry," Computer Vision and Image
#   Understanding, 2000.
#

    #retrieve parameters
    samples = parameters['samples']
    threshold = parameters['msac_thresh']
    trials = parameters['trials']
    flowDim = parameters['flow_dimensions']

    msacFit = functions['msac_fit']
    msacDist = functions['msac_dist']

    #get number of points
    noP = points.shape[0]

    #define worst case as initial guess
    bestCost = np.ones(flowDim)*threshold*noP
    bestInliers = np.zeros([noP, flowDim])

    #iterate through trials
    for i in np.arange(trials):

        #draw sample
        indx = np.random.permutation(noP)[0:samples]
        sample = points[indx,:]

        #fit to sample
        coeffs = msacFit(sample)
        #get residuals
        residuals = msacDist(coeffs, points)

        #compare to threshold
        residuals[residuals > threshold] = threshold
        inliers = residuals < threshold

        #cost
        cost = np.sum(residuals,0)
        compCost = bestCost > cost

        #adapt
        bestCost[compCost] = cost[compCost]
        bestInliers[:,compCost] = inliers[:,compCost]

    return bestCost, bestInliers
