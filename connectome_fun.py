""" Module with functions to facilitate working with the Drosophila connectome dataset
This module contains the following functions:
    - ax_colorcode:     color-code the tick labels of an imshow or pcolormesh plot
    - flatten:          flatten a list of lists
    - str_search:       search a list for a string
    - find_connected: 

Author:         Bjarne Schultze
Last modified:  01.12.2023
"""


import numpy as np
import pandas as pd
from matplotlib.offsetbox import (AnchoredOffsetbox,VPacker,TextArea)



def ax_colorcode(ax, color_code, c_names, *, legend=True, c_title):
    """ Color code the tick labels of an axes showing an imshow or pcolormesh plot

    Args:
        ax (axes handle):   axes containing the imshow/pcolormesh plot
        color_code (list):  the desired color for each tick label (length of list must match number of tick labels)
        c_cames (list):     names associated with each color (will be displayed in the legend)
        legend (bool):      (optional) indicating if a legend should be displayed, default: True
        c_title (str):      (optional) title for the legend

    Returns:
        None
    """
    # Check is number of unique colors matches the number of color names, raise error is not
    if len(set(color_code)) != len(c_names):
        raise ValueError("The number of colors in the color code does not match the number of given names.")

    # Color the axes tick labels according to the neurotransmitter type
    [ axi.set_color(col) for (col,axi) in zip(color_code, ax.xaxis.get_ticklabels())]
    [ axi.set_color(col) for (col,axi) in zip(color_code, ax.yaxis.get_ticklabels())]
    # Reduce the colors to a unique set
    colors = np.unique(np.array(color_code))

    # If requested, add a lengend for the tick colors
    if legend:
        # Create text boxes with the c_names written in the respective color
        boxes = []
        if c_title:
            # Add the title
            boxes.append(TextArea(c_title, textprops=dict(color="k", fontweight="bold", fontsize=12)))

        # Create one entry for each color using the given names (c_names)
        for i,prop in enumerate(c_names):
            boxes.append(TextArea(prop, textprops=dict(color=colors[i], fontsize=12)))

        # Stack the text boxes in one box
        box = VPacker(children=boxes, align="left", pad=0, sep=5)

        # Create legend box 
        anchored_box = AnchoredOffsetbox(loc='upper left', child=box, pad=0.5, frameon=True,
                                        bbox_to_anchor=(1.25, 1), bbox_transform=ax.transAxes, borderpad=0)
        # Add the legend to the right subplot
        ax.add_artist(anchored_box)



def flatten(l:list):
    """ Flatten list of lists

    Args:
        l (list): list to flatten
    Returns:
        flattend list
    """
    flat = []
    # Append non-list elements and extend list with list elements
    for el in l:
        if type(el) is list:
            flat.extend(flatten(el))
        else:
            flat.append(el)
    return flat



def str_search(s_text:str, l:list, *, logic_index=False):
    """ Search for text in a list
    
    Args:
        s_text (str):       text to search for
        l (list):           list to search in
        logic_index (bool): (optional), indicating whether to return logical indices or not, default: False
    
    Returns:
        indices of matching list elements or logic vector with same length as l
    """
    # Search list iteratively for text creating index or logical index list
    if logic_index:
        return [ True if e == s_text else False for _,e in enumerate(l) ]
    else: 
        return [ i for i,e in enumerate(l) if e == s_text ]



def find_connected(bodyIDs, connections, direction, neuron_ID_dict, *, normalization=True, collapse_by_type=False):
    """ Find the neurons that are connected to a given set of neurons
    Connections can be either upstream or downstream. The results can be combined per neuron type if requested. 

    Args:
        bodyIDs (int, list of int):     EM body IDs of the neurons for which the connections should be searched
        connections (pd.DataFrame):     manc traced connections table
        direction (str):                state whether to search for upstream "us" or downstream "ds" connections
        neuron_ID_dict (dict):          containing the neuron body IDs as keys to the neuron types
        normalization (bool):           (optional), state whether to normalize the weights (synapse count per connection) to the total synapse count of one neuron,
                                        default: True
        collapse_by_type (bool):        (optional), state whether to collapse the results by neuron type (True) or not (False), default: False
    
    Returns
        pd.DataFrame, containing the connected neurons and the weights for the connections
    """

    # Check the requested search direction
    if direction == "ds":
        direct = "bodyId_pre"
        catch_direct = "bodyId_post"
    elif direction == "us":
        direct = "bodyId_post"
        catch_direct = "bodyId_pre"
    else:
        raise ValueError("Invalid search direction. Must be 'us' or 'ds'.")

    # Get the type for each neuron
    type = np.array([ neuron_ID_dict.get(key) for key in bodyIDs ])

    # Allocate lists to store the connections
    conn_IDs = []
    conn_types = []
    conn_weights = []

    # Iterate over all given body IDs
    for cell in bodyIDs:
        # Find all connections for the current neuron and store their IDs
        find_conn = connections[direct] == cell
        conn_IDs.append(connections.loc[find_conn, catch_direct].to_numpy())

        # Find and store the type for all downstream neurons
        conn_types.append( [neuron_ID_dict.get(key) for key in conn_IDs[-1]] )

        # Collect the connection weights and normalize if requested
        if normalization and not collapse_by_type:
            # Normalize the weights to the summed weight (summed synapse count)
            norm_weights = connections.loc[find_conn, "weight"].to_numpy() / connections.loc[find_conn, "weight"].sum()
            conn_weights.append(norm_weights)
        else:
            conn_weights.append(connections.loc[find_conn, "weight"].to_numpy())


    # If the connections should be combined per cell type ...
    if collapse_by_type:
        typ_weights = []
        conn_utypes = []
        # For each neuron type among the given list of IDs ...
        for neuron in pd.unique(type):
            # Find all occurances of this type
            find_neuron = str_search(neuron, type)
            c_weights = flatten([ list(conn_weights[i]) for i in find_neuron ])
            # Sum the weights (synapse counts) over all neurons belonging to this type
            summed_weights = sum(c_weights)
            # Get the neuron types for each individual connection
            fl_types = np.array(flatten([ list(conn_types[i]) for i in find_neuron ]))

            typ_wght = []
            # For each type of connected neuron ...
            for typ in pd.unique(fl_types):
                # Find all occurances of this type of neuron
                find_typ = str_search(typ, fl_types)
                # Get the weights
                wghts = [ c_weights[i] for i in find_typ ]
                # Collect the summed weight for all neurons of this type (normalized if requested)
                if normalization:
                    typ_wght.append(sum(wghts) / summed_weights)
                else:
                    typ_wght.append(sum(wghts))

            # Collect the weights per neuron type and the types of the connected neurons
            typ_weights.append(typ_wght)
            conn_utypes.append(pd.unique(fl_types))

        # Combine the information in a data frame
        return pd.DataFrame({"type":pd.unique(type), "conn_type":conn_utypes, "conn_weight":typ_weights})
    else:
        # Combine all information in a data frame 
        return pd.DataFrame({"bodyId":bodyIDs, "type":type, "conn_bodyId":conn_IDs, "conn_type":conn_types, "conn_weight":conn_weights}).reset_index()
