# Author: Carola Fischer 
# Licence: CC-BY-NC
## imports
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from m_run import run
from m_plot_correction import plot_correction

## Set configuration
np.random.seed(274612)
run_4D = False #use 4D flow: yes or no
corr_fit_order = 3 #correction fit order, 0th-3rd order supported

## Load data
#load time-resolved examples / substitute by own code
#dimensions [dim1 dim2 slice time velocity]; for 2D data slice = 1;#
#type double
#example data: 2D flow AAo, 4D flow Aorta
###########################################################################
print('Load data ...')
if run_4D:
    phase_im_t = np.load('example_files/example4D_phase.npy')
    magnitude_im_t = np.load('example_files/example4D_magnitude.npy')
    venc = 150
else:
    phase_im_t = np.load('example_files/example2D_phase.npy')
    magnitude_im_t = np.load('example_files/example2D_magnitude.npy')
    venc = 150

## Prep data
#prep data to match requirements for msac and corr fits
#substitue by own code, but meet following requirements:
#   phase data ranges from -1 to 1 (# of venc)
#   magnitude data 2D scaled by peak value
#   magnitude data 4D scaled by peak value in center slice of phase encoding
print('Prep data ...')
phase_im_t = (phase_im_t/4096. - 0.5)*2 #scale to [-1,1[
if run_4D:
    #scale to center slice of Phase encoding direction (here 2nd)
    m,n,l,t = magnitude_im_t.shape
    center_slice = magnitude_im_t[:,int(np.ceil(n/2)),:,:]
    magnitude_im_t = magnitude_im_t/np.max(center_slice.flatten())
else:
    magnitude_im_t = magnitude_im_t/np.max(magnitude_im_t.flatten()) #scale to peak value

## Load configuration // parameters from publication
print('Load parameters ...')
parameters = dict()
parameters['msac_thresh'] = 0.01 #MSAC threshold in #venc
parameters['samples'] = 10 #MSAC sample size
parameters['trials'] = 100 #MSAC trials
parameters['msac_fit_order'] = 1 #linear fit during MSAC
if run_4D:
    parameters['mgn_thresh'] = 0.12 #magnitude mask threshold
    parameters['flow_dimensions'] = 3 # # of flow encoding directions
else:
    parameters['mgn_thresh'] = 0.08
    parameters['flow_dimensions'] = 1

parameters['corr_fit_order'] = corr_fit_order #correction fit order


## Run MSAC
print('Start correction ...')
results = run(phase_im_t, magnitude_im_t, parameters)
print(results['model'])

## Plot MSAC
plot_correction(phase_im_t, results, venc, run_4D)
