# PCMRI-MSAC
MSAC implementation to automatically correct background phase in PC-MRI images using python 3.8

## Licence
This code and data is published under the CC-BY-NC

## Citation
Please cite the corresponding paper to this code, if you reuse it for your own purposes.

Carola Fischer, Jens Wetzl, Tobias Schaeffter and Daniel Giese
**Fully automated background phase correction using M-estimate SAmple consensus (MSAC)—Application to 2D and 4D flow**
https://doi.org/10.1002/mrm.29363

## Prerequisites
python >= 3.8,
numpy >= 1.18.5,
matplotlib >= 3.2.2,

This code might work with other versions but this was not confirmed and is not guaranteed.
Only basic numpy and matplotlib.pyplot functions are used.

## How to use execute_MSAC.py
1. Define if 2D or 4D flow data is used
2. Choose polynomial fit order for final correction fit (0th-3rd order supported)
3. Write data loader or use given example
4. If wanted, play with MSAC parameters and fit orders


## Code structure
**Public:**
execute_MSAC.py:       Runs example code and can be modified to load own data  

**Private:**  
m_plot_correction.py:  Plots correction and MSAC mask   
m_run.py:              runs MSAC and corrects background phase from velocity data       
                     
                         
