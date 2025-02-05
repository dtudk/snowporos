# -*- coding: utf-8 -*-
"""
Copyright © 2025 Technical University of Denmark

*** Write here 
"""
#%% Import of the desired libraries from Python
import skimage 
import numpy as np
from skimage .morphology import dilation
from skimage .morphology import disk
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import porespy 
from porespy.tools import (
    Results)

#%% Definition of the function

def snow_porosimetry(snow,porosimetry,binary_image):
    """ 
    This funtion segmentates the porous space according to the SNOWPOROS algorithm. The aim is to reduce the oversegmentation of enlongated elliptical pores caused by marker-based watershed segmentation algorithms.
    The map obtained from the porosimetry map (or local thickness) is used to filter the markers obtained from the SNOW algortihm.Hence, only the peaks that appear inside
    the largest circles or spheres (compared to the circles that sourrounds the aforementioned circle) are selected. 
    
    The filtering is based on two major steps:
        
        -First filter: If two peaks from the SNOW algorithm are in the same circle (or two overlapped circle with the same size), only one is selected.This the one which have the biggest value in the Euclidean distance map
        -Second filter: Only the peaks that are inside the largest discs are selected. Instead with every pixel,
        it is done with every region from the labeled region. 
    
    Aftewards, the new filters are passed to normal marker-based watershed segmentation (found in Python package skimage.segmentation.watershed) to perform the new segmention of the pore space. 
    
    
    Inputs parameters:
    ---------
        -snow= results from the SNOW algorithm. distance map, maximum peaks, labelled regions.
        -porosimetry: Map obtained from the porosimetry method.
        -binary image: This is the binary image of the porous structure.
    
    Return parameter
    ---------
    results: An object previously designed by Porespy´s authors. The object has the following attributes:
        
    Attribute
    
    regions: The void space segmented according to the SNOWPOROS algorithm differents units (marked as labels).
    markers: Boolean image/matrix where the pixels with value 1 denote the place where the SNOWPOROS algorithm places the selected markers. 

    
   This function can be called and applied similarly to the SNOW algorithm. Vist the following page for inspiration:
    ----------
    
   [1] https://porespy.org/examples/filters/reference/snow_partitioning.html
    
    
    Important references to read
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm using marker-based watershed segmentation".  Physical Review E. (2017)
    
    """
    # Labelling the porosimetry map. All the pixels with the same value in the porosimetry map and are connected are labelled with the same value.
    #It is necessary for futures processess

    labels_poros=skimage.measure.label(10000*porosimetry) #Multiply by 1000 for differentiating better for the labelling, which only recognized int values
    
    # Labelling the position of the maxima points calculated by the SNOW algorithm. The position of the peaks are obtained in an array vector
    
    coord_peak_snow=peak_local_max(snow.peaks,footprint=np.ones((1, 1)), labels=binary_image)
    
    #Organization of the data. Define the array for the peaks:  column 0=row position, column 1: Columm position , column_2: label of the porosimetry,column 3:value from the distance map).
    
           # Each row from this vector represents a maxima point
    
    new_coord=np.zeros((coord_peak_snow.shape[0],4),dtype=int) # First they are zeros
    
            # Incorporation of the position
    
    new_coord[:,0]=coord_peak_snow[:,0]
    new_coord[:,1]=coord_peak_snow[:,1]
   
    snow.dt=snow.dt.astype(int)
    
            # Each position is linked to: Labels from the porosimetry and the value of the Euclidean distance map (obtained thanks to the SNOW algorithm) 
    for i in range(new_coord.shape[0]):
        new_coord[i,2]=labels_poros[new_coord[i,0],new_coord[i,1]]
        new_coord[i,3]=snow.dt[new_coord[i,0],new_coord[i,1]]
    
    
    # First filter
    
            # Creation of a matrix which only contains unique values of the labels
    
    snow_poros_unique=np.unique(new_coord[:,2]) # This a vector that only contains unique values of the labels from the porosimetry.
    
    new_coords_corrected=np.zeros((snow_poros_unique.shape[0],4),dtype=int) # Here is where I am going to include the labels.
    
    j=0 # Counting number for the row of the matrix to fill. 

    for region in snow_poros_unique: 
        
        indices=np.where(new_coord[:,2]==region) # Where is the label in the new coord. It gives one or several points. 
        #The indices show the row value in the new_coord matrix
        values=new_coord[indices,3] #Value in the distance map
        array_indices=np.array(indices) # Transform to tuple to an array 
        array_values=np.array(values) # Transform value of the value into an array.
        row=array_indices[0 ,np.argmax(array_values)] # Value of the row with the highest value in the distance map.
        new_coords_corrected[j,:]=new_coord[row,:] # Filling the new matrix with the corrected coordinates
        j= j+1
        
    # Second filter
    coordenate_change=new_coords_corrected[:,:2].copy() 
    
    labelled_poros=np.zeros((len(coordenate_change),2))# These are values of peaks: column_0:label from the labelling on on the porosimetry,column_1: value in the porosimetry map
    # We did this to have in the float form.

    
    for i in range(len(coordenate_change)):
        labelled_poros[i,0]=labels_poros[coordenate_change[i,0],coordenate_change[i,1]].astype(np.float64) # Value of the label
        labelled_poros[i,1]=porosimetry[coordenate_change[i,0],coordenate_change[i,1]]  # Value of the pixel in the porosimetry map
        
    coordenates_true_peaks =np.zeros((1,2))
    
    for i in range(len(coordenate_change)):
        
        label_value= labelled_poros[i,0]  # Label of the region selected.
        mask=labels_poros==label_value # Mask of the desired region. This means I only consider one label in the labelled map. True 
        #value is given to the desired regions. False value is given to the rest.
        
        dilated_mask=dilation(mask,disk(1)) # Dilation of one region
        porosimetry_mask1=porosimetry.copy()  
        porosimetry_mask2=porosimetry.copy()
        
        # Application of the mask
        porosimetry_mask1[dilated_mask==0]=0 
        porosimetry_mask2[dilated_mask==0]=0
        
        # Change the porosimetry_1 of the mask with the value of the center-
        porosimetry_mask1[dilated_mask!=0]=labelled_poros[i,1]  # We only the value of before.
        
        # Difference between the two values (this pseudo -dilation)
        diff=porosimetry_mask1-porosimetry_mask2
        
        if np.min(diff)>=0:  # This is the condition to be a real peak
            true_peak=np.zeros((1,2))
            true_peak[0,0] =coordenate_change[i,0]
            true_peak[0,1] = coordenate_change[i,1]
            coordenates_true_peaks=np.r_[coordenates_true_peaks,true_peak]
   
    #Transform the peaks into a int. Coordenates_true_peaks shows the posiw and column position) of the new 
    coordenates_true_peaks=coordenates_true_peaks.astype(np.int32)
      
    #Perfom the final marker-based watershed segmentation with the new position of the seeds or markers (the variable is coordinates_true_peaks)
    mask_2_filt = np.zeros(binary_image.shape, dtype=bool) 
    mask_2_filt[tuple(coordenates_true_peaks[:,:2].T)] = True # Binary image where the markers are markers as "True". The rest of the pixels are denoted as "False". 
    markers, _ = ndi.label(mask_2_filt) 
    labels_2_filt = watershed(-snow.dt, markers, mask=binary_image) 
    
       #Labelling the tiny pores which could not see by the SNOW algorithm
       #Preparing the image. Only show the tiny pores that could not be seen by the SNOW algorithm
    binary_image_copy=binary_image.copy()
    binary_image_copy[labels_2_filt>0]=0

       #Labelling the small pores
    label_small_pores= skimage.measure.label(binary_image_copy,connectivity=2)

        #Labelling small pores
    new_label_small_pores=label_small_pores + np.max(labels_2_filt)*(binary_image_copy>0)
    new_label_small_pores_int=new_label_small_pores.astype(int)  
    
    # Final segmented image. Each pore is designated with a label/number
    final_label=new_label_small_pores_int+labels_2_filt
    
    #Organization of the final data
    Results
    tup=Results()
    tup.regions=final_label*(binary_image>0)
    tup.markers=mask_2_filt
    
    return tup
