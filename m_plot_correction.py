# Licence: CC-BY-NC
# Author: Carola Fischer

import numpy as np
import matplotlib.pyplot as plt


def plot_correction(phase_im_t, results, venc, run_4D):
# Plotting routine to display outlier sensitive stationary tissue masks
# for PC-MRI 2D or 4D flow data
# 4D flow:  each velocity encoding dimension is plotted in a new figure
#
# Input:
# -----
# phase_im_t   - time-resolved phase data [-1,1[
#                dimensions: [dim1 dim2 slice time velEnc]
#
# results      - struct containing the results from m_run
#
# venc         - maximum encoded velocity in phase_im_t
#
# run_4D       - boolean to distinguish between 2D or 4D flow
#


    #scale bgr to 8%venc
    bgr_scaled = results['bgr_msac'].copy()*venc
    bgr_scaled[bgr_scaled < -10] = -10
    bgr_scaled[bgr_scaled >  10] =  10

    #scale corr_avg to 8%venc
    cavg_scaled = results['corr_avg'].copy()*venc
    cavg_scaled[cavg_scaled < -10] = -10
    cavg_scaled[cavg_scaled >  10] =  10

    #scale uncorrected phase average to 10cm/s
    avg_phase = np.squeeze(np.mean(phase_im_t,3))
    avg_scaled = avg_phase.copy()*venc
    avg_scaled[avg_scaled < -10] = -10
    avg_scaled[avg_scaled >  10] =  10

    m,n,l,t = phase_im_t.shape[0:4]
    #define no of velocity directions and which slice is plotted
    if run_4D:
        d = 3
        slice = int(np.ceil(l/2)) #or any other slice
    else:
        d = 1
        slice = 1


    #plot
    if run_4D:
        direction = ['left-right', 'anterior-posterior', 'head-foot']
    else:
        direction = ['through-plane']
    for dir in np.arange(d):
        fig = plt.figure()
        fig.suptitle('Results: velocity dim %s'%(direction[dir]))

        if run_4D:
            avg_s = np.squeeze(avg_scaled[:,:,slice,dir])
            avg_p = np.squeeze(avg_phase[:,:,slice,dir])*venc
            cavg_s = np.squeeze(cavg_scaled[:,:,slice,dir])
            mask_mgn = np.squeeze(results['mask_mgn'][:,:,slice])
            mask_msac = np.squeeze(results['mask_msac'][:,:,slice,dir])
            corr_avg = np.squeeze(results['corr_avg'][:,:,slice,dir])*venc
        else:
            avg_s = np.squeeze(avg_scaled[:,:])
            avg_p = np.squeeze(avg_phase[:,:])*venc
            cavg_s = np.squeeze(cavg_scaled[:,:])
            mask_mgn = np.squeeze(results['mask_mgn'][:,:])
            mask_msac = np.squeeze(results['mask_msac'][:,:])
            corr_avg = np.squeeze(results['corr_avg'][:,:])*venc


        axes = []
        ims = []
        ax1 = plt.subplot2grid((2,3), (0,0))
        ims.append(ax1.imshow(avg_s, cmap = 'RdBu'))# colorbar xticks([]) yticks([])
        ax1.set_title('Velocity - uncorrected')
        axes.append(ax1)

        ax2 = plt.subplot2grid((2,3), (1,0))
        ims.append(ax2.imshow(cavg_s, cmap = 'RdBu'))# colorbar xticks([]) yticks([])
        ax2.set_title('Velocity - corrected')
        axes.append(ax2)

        ax3 = plt.subplot2grid((2,3), (0,1))
        ims.append(ax3.imshow(mask_mgn, cmap = 'gray'))# colorbar xticks([]) yticks([])
        ax3.set_title('Input MSAC, Slice %d'%slice)
        axes.append(ax3)

        ax4 = plt.subplot2grid((2,3), (1,1))
        ims.append(ax4.imshow(mask_msac, cmap = 'gray'))
        ax4.set_title('Output MSAC')
        axes.append(ax4)

        ax5 = plt.subplot2grid((2,3), (0,2), rowspan=2)
        ax5.hist(avg_p[mask_msac == 1], 150, alpha = 0.7)
        ax5.hist(corr_avg[mask_msac == 1], 150, alpha = 0.7)
        ax5.set_xlabel('Velocity [cm/s]')
        ax5.set_ylabel('Count')
        ax5.legend(['Before correction', 'After correction'])
        ax5.set_title('Vel distribution within mgn mask')
        axes.append(ax5)

        for axi in axes[0:4]:
            axi.set_xticks([])
            axi.set_yticks([])

        fig.colorbar(ims[0], ax=axes[0])
        fig.colorbar(ims[1], ax=axes[1])
        # colormap(ax(1), cmap) colormap(ax(2), cmap)
        # colormap(ax(3), 'gray') colormap(ax(4), 'gray')

    plt.show()
