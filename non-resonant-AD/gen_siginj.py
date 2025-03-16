import argparse
import numpy as np
import pickle
import os
from helpers.process_data import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigsample",help="Input signal .txt file",
    default="/home/aegis/Artemis/NRAD/working_dir/samples/sig_samples/SVJL.root.txt"
)
parser.add_argument("-b1","--bkg-dir",help="Input bkground folder",
    default="/home/aegis/Artemis/NRAD/working_dir/samples/qcd_data_samples/"
)
parser.add_argument("-size",type=int,help="Number of bkg text files",default=20)
parser.add_argument("-o","--outdir",help="output directory", default ="/home/aegis/Titan0/NRAD/SPP_NRAD")
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)

args = parser.parse_args()

def main():

    # Create the output directory
    data_dir = f"{args.outdir}/data/"
    os.makedirs(data_dir, exist_ok=True)
    
    # define sample size as the number of files
    sample_size = args.size

    var_names = ["ht", "met", "m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    
    # First load in signal
    variables, sig = load_samples(args.sigsample)
    sig = get_quality_events(sig)
    sig_events = sort_event_arr(var_names, variables, sig)
    
    with open(f"{data_dir}/mc_scaler.pkl","rb") as f:
        print("Loading in trained minmax scaler.")
        scaler = pickle.load(f)
    
    print(f"Loading {sample_size} samples of bkg")
    
    bkg_events_list = []
    for i in range(sample_size):
        bkg_path = f"{args.bkg_dir}/qcd_{i}.txt"
        if os.path.isfile(bkg_path):  
            # Load input events ordered by varibles
            _, bkg_i = load_samples(bkg_path)
            bkg_i = get_quality_events(bkg_i)
            bkg_events_list.append(sort_event_arr(var_names, variables, bkg_i))
        else:
            check_file_log(bkg_pathh)
    if len(bkg_events_list)==0:
        sys.exit("No files loaded. Exit...")
        
    print("Done loading!")
    
    # concatenate all background events
    bkg_events = np.concatenate(bkg_events_list)

    # SR masks
    bkg_mask_SR = phys_SR_mask(bkg_events)
    bkg_mask_CR = ~bkg_mask_SR
    
    # Create folder for the particular signal injection
    seeded_data_dir = f"{data_dir}/seed{args.gen_seed}/"
    os.makedirs(seeded_data_dir, exist_ok=True)
    np.random.seed(int(args.gen_seed))
    
    sig_percent_list = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024]
    
    # Create signal injection dataset
    n_bkg_SR = bkg_events[bkg_mask_SR].shape[0]
    
    for s in sig_percent_list:
        n_sig = int(s*n_bkg_SR) #Number of signal depends on the background events in SR mask
        selected_sig_indices = np.random.choice(sig_events.shape[0], size = n_sig, replace = False)
        selected_sig = sig_events[selected_sig_indices, :]
        data_events = np.concatenate([selected_sig, bkg_events])
        
        #SR masks
        data_mask_SR = phys_SR_mask(data_events)
        data_mask_CR = ~data_mask_SR
        
        selected_sig_mask_SR = phys_SR_mask(selected_sig)
        
        sig_list = selected_sig[selected_sig_mask_SR]
        bkg_list = bkg_events[bkg_mask_SR]
        
        #SR Events
        n_sig_SR = selected_sig[selected_sig_mask_SR].shape[0]
        s_SR = round(n_sig_SR/n_bkg_SR, 5)
        significance = round(n_sig_SR/np.sqrt(n_bkg_SR), 5)
        
        #Print Dataset Information
        print(f"S/B={s_SR} in SR, S/sqrt(B) = {significance}, N bkg SR: {n_bkg_SR:.1e}, N sig SR: {n_sig_SR}")
        
        # Save dataset
        np.savez(f"{seeded_data_dir}/data_{s}.npz", data_events_cr=scaler.transform(data_events[data_mask_CR]), data_events_sr=scaler.transform(data_events[data_mask_SR]), selected_sigs_sr=selected_sig[selected_sig_mask_SR], selected_sigs_cr=selected_sig[~selected_sig_mask_SR], bkg_events_SR =bkg_events[bkg_mask_SR], bkg_events_CR=bkg_events[bkg_mask_CR], sig_percent=s_SR, signif = significance)
        
    print(f"Finished generating dataset. (Gen seed: {args.gen_seed})")

if __name__ == "__main__":
    main()
    
    
