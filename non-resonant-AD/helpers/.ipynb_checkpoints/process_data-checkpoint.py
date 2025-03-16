import numpy as np
import sys
import os

# Assume all events has the structure of the following:
# events[:, 0] is context variable 1
# events[:, 1] is context variable 2
# events[:, 1:] are the feature variables



def phys_SR_mask(events):

    # define SR and CR masks
    HT_cut = 800    # In SR, HT > 800 GeV
    MET_cut = 75    # In SR, MET > 75 GeV

    # SR masks
    if events.shape[1]>1:
        mask_SR = (events[:, 0] > HT_cut) & (events[:, 1] > MET_cut)
        return mask_SR
    else:
        sys.exit(f"Wrong input events array. Array dim {events.shape[1]}, must be >= 2. Exiting...")
        

def get_quality_events(arr):

    if np.isnan(arr).any():
        return arr[~np.isnan(arr).any(axis=1)]
    
    else:
        return arr

def check_file_log(bkg_path = None, ideal_bkg_path = None, mc_path = None):

    for file_path in [bkg_path, ideal_bkg_path, mc_path]:
        if file_path != None:
            if not os.path.isfile(file_path):
                print(f"{file_path} does not exist!")
             

def morph_mc(mc_events):
    """
    This function has been hand-tuned to samples in the official Zenodo dataset. Be aware!!
    """
    morphed_mc_events = np.copy(mc_events)

    def morph_ht(x):
        return x

    def morph_met(x):
        return (x*(1+(x/500.)))

    def morph_mjj(x):
        return x*(1+(x/6000.))

    def morph_taus(x):
        return x*(x**(0.1))

    morphed_mc_events[:,0] = morph_ht(morphed_mc_events[:,0])
    morphed_mc_events[:,1] = morph_met(morphed_mc_events[:,1])
    morphed_mc_events[:,2] = morph_mjj(morphed_mc_events[:,2])
    morphed_mc_events[:,3] = morph_taus(morphed_mc_events[:,3])
    morphed_mc_events[:,4] = morph_taus(morphed_mc_events[:,4])
    morphed_mc_events[:,5] = morph_taus(morphed_mc_events[:,5])
    morphed_mc_events[:,6] = morph_taus(morphed_mc_events[:,6])

    return morphed_mc_events

#From Semi-visible jet utils
def load_samples(file):
    samples = np.loadtxt(file, dtype=str)
    # Get the names of all varibles
    variables = samples[0]
    # Get the events ordered by varibles
    events = np.asarray(samples[1:], dtype = float)
    return variables, events

def sort_event_arr(names, variables, events):
    
    event_list = []
    
    for x in names:   
        ind_x = ind(variables, x)
        event_list.append(events[:, ind_x])
    
    return np.stack(event_list, axis=1)

def ind(variables, name):
    return np.where(variables == name)[0][0]