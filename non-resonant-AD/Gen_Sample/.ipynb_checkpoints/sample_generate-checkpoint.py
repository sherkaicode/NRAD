import torch
import os
import sys
import argparse

import numpy as np
script_dir = os.path.dirname(__file__) 
helpers_path = os.path.join(script_dir, '..', 'model_scripts')  
sys.path.insert(0, os.path.abspath(helpers_path))
from SimpleMAF import SimpleMAF


parser = argparse.ArgumentParser()
parser.add_argument("-i","--indir",help="working folder",default="/home/aegis/Titan0/NRAD/SPP_NRAD")
parser.add_argument("-s","--signal",default=None,help="signal fraction",)
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument("-o","--oversample",help="How much to oversample the model",default=1)
parser.add_argument( "-v", "--verbose",default=False,help="Verbose enable DEBUG")
#parser.add_argument("-cu", "--cuda_slot", help = "cuda_slot")

args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)
def main():
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    #device = torch.device("cpu")
    data_path = f"{args.indir}/data/seed{args.gen_seed}"
    samples_path = f"{args.indir}/samples/seed{args.gen_seed}"
    model_path = f"{args.indir}/models/seed{args.gen_seed}"
    os.makedirs(samples_path, exist_ok=True)
    
    mc_path = f"{args.indir}/data/mc_events.npz"
    mc_events = np.load(mc_path)
    mc_events_sr = mc_events["mc_events_sr"]
    
    n_withold = 10000 
    n_context = 2
    n_features = 5
    
    per = []
    if args.signal is not None:
        per = [args.signal]
    else:
        per = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024]
    
    for p in per:
        #Load Context for generate
        data_events = np.load(f"{data_path}/data_{p}.npz")
        data_events_cr = data_events["data_events_cr"]
        data_events_sr = data_events["data_events_sr"]
        data_context_cr_test = data_events_cr[-n_withold:,:n_context]
        data_feature_cr_test = data_events_cr[-n_withold:,n_context:]
        data_feature_sr_test = data_events_sr[-n_withold:,n_context:]
        mc_context_sr = mc_events_sr[:,:n_context]
        
        #Load Model
        if os.path.isfile(f"{model_path}/generate_{p}.pt"):
            print("Loading model")
            MAF = torch.load(f"{model_path}/generate_{p}.pt")
            MAF.to(device)
            
        print("Making samples for s/b =", p)
        # CR Background Predictions
        pred_bkg_CR = MAF.sample(1, data_context_cr_test)
        np.savez(f"{samples_path}/generate_CR_samples_{p}.npz", target_cr=data_feature_cr_test, generate_cr=pred_bkg_CR)
        
        # SR Background Predictions
        pred_bkg_SR = MAF.sample(args.oversample, mc_context_sr)
        np.savez(f"{samples_path}/generate_SR_samples_{p}.npz", data_sr=data_feature_sr_test,samples = pred_bkg_SR)
        
    
if __name__ == "__main__":
    main()
