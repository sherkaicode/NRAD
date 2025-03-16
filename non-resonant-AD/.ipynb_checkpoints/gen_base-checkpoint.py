# This script generate the Test Dataset and the scaler for preprocessing

import argparse
import numpy as np
import pickle
import os
from helpers.process_data import *
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument( "-s", "--sigsample",help="Input signal .txt file",
    default="/home/aegis/Artemis/NRAD/working_dir/samples/sig_samples/SVJL.root.txt"
)
parser.add_argument("-t","--test-dir",help="Input bkground folder",
    default="/home/aegis/Artemis/NRAD/working_dir/samples/qcd_test_samples/" 
)
parser.add_argument("-mc", "--mc-dir",help="Input MC bkground folder",
    default="/home/aegis/Artemis/NRAD/working_dir/samples/qcd_mc_samples/" 
)
parser.add_argument("-stest", type=int,help="Number of bkg text files",default=46)
parser.add_argument("-smc", type=int, help="Number of MC text files", default=20)
parser.add_argument("-o","--outdir",help="output directory", default = "/home/aegis/Titan0/NRAD/SPP_NRAD")
parser.add_argument("-morph_mc",action='store_true',help="Whether to tamper with the MC to make it look more different from the data")
args = parser.parse_args()

def main():

    # Create the output directory
    data_dir = f"{args.outdir}/data/"
    os.makedirs(data_dir, exist_ok=True)
    
    # define sample size as the number of files
    stest = args.stest
    smc = args.smc
        
    print(f"Loading {stest} Test samples and {smc} MC samples...")
    
    # load signal first
    var_names = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    variables, sig = load_samples(args.sigsample)
    sig = get_quality_events(sig)
    sig_events = sort_event_arr(var_names, variables, sig)

    test_events_list = []
    mc_events_list = []
    for i in range(max(stest, smc)):
        # Load input events ordered by varibles
        
        if i < stest:
            test_path = f"{args.test_dir}/qcd_{i}.txt"
            if os.path.isfile(test_path):
                _, test_i = load_samples(test_path)
                test_i = get_quality_events(test_i)
                test_events_list.append(sort_event_arr(var_names, variables, test_i))
            else:
                check_file_log(bkg_path=test_path)
        if i < smc:
            mc_path = f"{args.mc_dir}/qcd_{i}.txt"
            if os.path.isfile(mc_path):
                _, mc_i = load_samples(mc_path)
                mc_i = get_quality_events(mc_i)
                mc_events_list.append(sort_event_arr(var_names, variables, mc_i))
            else:
                check_file_log(mc_path=mc_path)
            
    if (len(test_events_list)==0) or (len(mc_events_list)==0):
            sys.exit("No files loaded. Exit...")
    
    print("Done loading!")
    
    # concatenate all backgroud events
    test_events = np.concatenate(test_events_list)
    mc_events = np.concatenate(mc_events_list)
    
    if args.morph_mc:
        print("Morphing the mc events a bit...")
        mc_events = morph_mc(mc_events)
    print(mc_events.shape)
    # preprocess data -- fit to MC
    scaler = preprocessing.MinMaxScaler(feature_range=(-2.5, 2.5)).fit(mc_events)
    
    with open(f"{data_dir}/mc_scaler.pkl","wb") as f:
        print("Saving out trained minmax scaler.")
        pickle.dump(scaler, f)
    
    # SR Cuts
    test_mask_SR = phys_SR_mask(test_events)
    test_events_SR = test_events[test_mask_SR]
    mc_mask_SR = phys_SR_mask(mc_events)
    mc_events_SR = mc_events[mc_mask_SR]
    
    # CR Cuts
    mc_mask_CR = ~mc_mask_SR
    mc_events_CR = mc_events[mc_mask_CR]
    
    sig_mask_SR = phys_SR_mask(sig_events)
    sig_events_SR = sig_events[sig_mask_SR]

    #Save MC static dataset
    
    np.savez(f"{data_dir}/mc_events.npz", mc_events_cr=scaler.transform(mc_events[mc_mask_CR]), mc_events_sr=scaler.transform(mc_events[mc_mask_SR]))
    
    print(test_events.shape)
    print(test_events_SR.shape)
    # Select test set
    n_test_sig = 30276
    n_test_bkg = 121103
    sig_test_SR = sig_events_SR[:n_test_sig]
    bkg_test_SR = test_events_SR[:n_test_bkg]
    
    # Select fully supervised set
    sig_fullsup_SR = sig_events_SR[n_test_sig:(n_test_sig + 1770)]
    bkg_fullsup_SR = test_events_SR[n_test_bkg:]
   
    print(f"Test dataset in SR: N sig={len(sig_test_SR)}, N bkg={len(bkg_test_SR)}")
    print(f"Fully supervised dataset in SR: N sig={len(sig_fullsup_SR)}, N bkg={len(bkg_fullsup_SR)}")

    # Save dataset
    np.savez(f"{data_dir}/test_SR.npz", bkg_events_SR=scaler.transform(bkg_test_SR), sig_events_SR=scaler.transform(sig_test_SR))
    np.savez(f"{data_dir}/fullsup_SR.npz", bkg_events_SR=scaler.transform(bkg_fullsup_SR), sig_events_SR=scaler.transform(sig_fullsup_SR))
            
    print(f"Finished generating datasets.")

if __name__ == "__main__":
    main()
