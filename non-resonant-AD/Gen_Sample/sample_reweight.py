import torch
import os
import sys
import argparse
import numpy as np

script_dir = os.path.dirname(__file__) 
helpers_path = os.path.join(script_dir, '..', 'model_scripts')  
sys.path.insert(0, os.path.abspath(helpers_path))
from Classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument("-i","--indir",help="working folder",default="/home/aegis/Titan0/NRAD/SPP_NRAD")
parser.add_argument("-s","--signal",default=None,help="signal fraction",)
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument( "-v", "--verbose",default=False,help="Verbose enable DEBUG")
#parser.add_argument("-cu", "--cuda_slot", help = "cuda_slot")

args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)

# logging.basicConfig(level=logging.INFO)
# log_level = logging.DEBUG if args.verbose else logging.INFO
# log = logging.getLogger("run")
# log.setLevel(log_level)

def main():
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    data_path = f"{args.indir}/data/seed{args.gen_seed}"
    samples_path = f"{args.indir}/samples/seed{args.gen_seed}"
    model_path = f"{args.indir}/models/seed{args.gen_seed}"
    os.makedirs(samples_path, exist_ok=True)
    
    mc_path = f"{args.indir}/data/mc_events.npz"
    mc_events = np.load(mc_path)
    mc_events_cr = mc_events["mc_events_cr"]
    mc_events_sr = mc_events["mc_events_sr"]
    
    n_withold = 10000 
    mc_cr_test = mc_events_cr[-n_withold:]
    
    per = []
    if args.signal is not None:
        per = [args.signal]
    else:
        per = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024]
    
    for p in per:
        #Load Data
        data_events = np.load(f"{data_path}/data_{p}.npz")
        data_events_cr = data_events["data_events_cr"]
        data_events_sr = data_events["data_events_sr"]
        
        data_cr_test = data_events_cr[-n_withold:]
        data_sr_test = data_events_sr[-n_withold:]
        mc_cr_test = mc_events_cr[-n_withold:]
        
        #Load Model
        if os.path.isfile(f"{model_path}/reweight_{p}.pt"):
            print("Loading model")
            NN_reweight = torch.load(f"{model_path}/reweight_{p}.pt")
            NN_reweight.to(device)
        
        print("Making samples for s/b =", p)
        # CR Samples weights
        w_cr = NN_reweight.evaluation(mc_cr_test)
        w_cr = (w_cr/(1.-w_cr)).flatten()
        np.savez(f"{samples_path}/reweight_CR_samples_{p}.npz", target_cr=data_cr_test, mc_cr=mc_cr_test, w_cr=np.nan_to_num(w_cr, copy=False, nan=0.0, posinf=0.0, neginf=0.0))
        
        # SR Samples weights
        w_sr = NN_reweight.evaluation(mc_events_sr)
        w_sr = (w_sr/(1.-w_sr)).flatten()
        np.savez(f"{samples_path}/reweight_SR_samples_{p}.npz", data_sr=data_sr_test, mc_samples=mc_events_sr, w_sr=np.nan_to_num(w_sr, copy=False, nan=0.0, posinf=0.0, neginf=0.0))
        
        print("Generated Samples for s/b =", p)

if __name__ == "__main__":
    main()
    

        
        
