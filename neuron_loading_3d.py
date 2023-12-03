''' Module with functions for retrieving 3D reconstructions for neurons and regions of interest in the MANC dataset
This module contains the following functions:
    - load_neurons:     load 3D neuron reconstructions from hard drive or download them to the hard drive before loading
    - load_rois:        load 3D reconstructions of regions of interest (neuromeres and neuropils)

Author: Bjarne Schultze
Last modified: 01.12.2023
'''

import os
import numpy as np
# Re-create deprecated aliases which are necessary for the navis module
np.int = np.int_
np.bool = np.bool_
import navis.interfaces.neuprint as neu
import navis
import cloudvolume as cv

# This needs to be run only once at the beginning of each session (ensures that cloudvolume gives back navis neurons)
navis.patch_cloudvolume()



def load_neurons(neuronIDs:list, neuron_ID_dict:dict):
    ''' Load 3D neuron reconstrucitons (volumes)

    Args:
        neuronIDs (list): neuron body IDs of the neuron volumes to load
        neuron_ID_dict (dict): matching the neuron body IDs to cell type names {bodyID: 'cell_type', ...}
    Retruns:
        navis NeuronList (similar to pd.DataFrame) with neuron volumes and meta data
    '''

    # Download the neuron 3D reconstructions or load them if they are saved locally
    try:
        if len(os.listdir("./neuron_volumes/")) == len(neuronIDs):
            # Load the existing data
            mn_meshes = navis.read_mesh("./neuron_volumes/*.obj")
            # Add the readable neuron types for plotting
            for cell in range(len(mn_meshes)):
                mn_meshes[cell].name = f"{neuron_ID_dict[int(mn_meshes[cell].name)]}"
        else:
            raise('There are less .obj files than requested by neuronIDs.')
    except:
        # Initialize connection to the manc dataset via CloudVolume
        vol = cv.CloudVolume('precomputed://gs://manc-seg-2022-05/manc-seg-2022-05', use_https=True, progress=False)
        # Get the mesh data for all wing motor neurons (level of detail 3, 1 for highest resolution)
        mn_meshes = vol.mesh.get(neuronIDs, as_navis=True, lod=3)
        # Add the readable neuron types for plotting
        for cell in range(len(mn_meshes)):
            mn_meshes[cell].name = neuron_ID_dict[neuronIDs[cell]]
            mn_meshes[cell].id = neuronIDs[cell]

        # Store the neuron meshes locally
        os.makedirs('./neuron_volumes/', exist_ok=True)
        navis.write_mesh(mn_meshes, filepath="./neuron_volumes/", filetype='obj')
    
    return mn_meshes



def load_rois(roiIDs:list, roi_ID_dict:dict):
    ''' Load 3D reconstructed regoins of interest (rois, neuropils/neuromeres)

    Args:
        roiIDs (list): IDs of the regions of interest
        roi_ID_dict (dict): matching the roi IDs to readable names {roiID: 'roi_name', ...}
    Retruns:
        navis NeuronList (similar to pd.DataFrame) with roi volumes and metadata
    '''
    try:
        # Download the neuron 3D reconstructions or load them if they are saved locally
        if len(os.listdir("./roi_volumes/")) == len(roiIDs):
            # Load the existing data
            roi_meshes = navis.read_mesh("./roi_volumes/*.obj")
            # Add the readable names for plotting
            for roi in range(len(roi_meshes)):
                roi_meshes[roi].name = roi_ID_dict[roiIDs[roi]]
        else:
            raise('There are less .obj files than requested by roiIDs.')

    except:
        # Initialize a new connection to the flyem dataset via CloudVolume
        vol_roi = cv.CloudVolume('precomputed://gs://flyem-vnc-roi-d5f392696f7a48e27f49fa1a9db5ee3b/roi', use_https=True, progress=False)
        # Get all desired ROIs and store them in a NeuronList
        roi_meshes = navis.NeuronList([ vol_roi.mesh.get(roi, as_navis=True) for roi in roiIDs ])
        # Add the readable names for plotting
        for roi in range(len(roi_meshes)):
            roi_meshes[roi].name = roi_ID_dict[roiIDs[roi]]
            roi_meshes[roi].id = roiIDs[roi]

        # Store the neuron meshes locally
        os.makedirs('./roi_volumes/', exist_ok=True)
        navis.write_mesh(roi_meshes, filepath="./roi_volumes/", filetype='obj')

    return roi_meshes