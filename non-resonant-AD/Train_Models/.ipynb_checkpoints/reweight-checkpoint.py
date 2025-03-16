import argparse
import numpy as np
import logging
import torch
import yaml
import sys
import os

script_dir = os.path.dirname(__file__) 
helpers_path = os.path.join(script_dir, '..', 'model_scripts')  
sys.path.insert(0, os.path.abspath(helpers_path))
from Classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument("-i","--indir",help="working folder",
    default="/home/aegis/Titan0/NRAD/SPP_NRAD"
)
parser.add_argument("-s","--signal",default=None,help="signal fraction",)
parser.add_argument("-c","--config",help="Reweight NN config file",default="configs/reweight_physics.yml")
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument( "-v", "--verbose",default=False,help="Verbose enable DEBUG")
#parser.add_argument("-cu", "--cuda_slot", help = "cuda_slot")
args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)

logging.basicConfig(level=logging.INFO)
log_level = logging.DEBUG if args.verbose else logging.INFO
log = logging.getLogger("run")
log.setLevel(log_level)

def main():
    Testing = False
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    seed_path = f"{args.indir}/data/seed{args.gen_seed}/"
    model_dir = f"{args.indir}/models/seed{args.gen_seed}/"
    os.makedirs(model_dir, exist_ok=True)
    
    mc_path = f"{args.indir}/data/mc_events.npz"
    mc_events = np.load(mc_path)
    mc_events_cr = mc_events["mc_events_cr"]
#     mc_events_sr = mc_events["mc_events_sr"]
    
    per = []
    if args.signal is not None:
        per = [args.signal]
    else:
        per = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024]
    for p in per:
        print("Working with s/b =", p)
        #Load Data
        data_events = np.load(f"{seed_path}/data_{p}.npz")
        data_events_cr = data_events["data_events_cr"]
        
        print("CR has", len(data_events_cr), "data events,", len(mc_events_cr), "MC events.")
        
        n_withold = 10000 
        
        data_cr_train = data_events_cr[:-n_withold]
        data_cr_test = data_events_cr[-n_withold:]
        mc_cr_train = mc_events_cr[:-n_withold]
        mc_cr_test = mc_events_cr[-n_withold:]
        
        
        input_x_train_CR = np.concatenate([mc_cr_train, data_cr_train], axis=0)
        
        #Labels for Classifier (1 for Data and 0 for MC)
        mc_cr_label = np.zeros(mc_cr_train.shape[0]).reshape(-1,1)
        data_cr_label = np.ones(data_cr_train.shape[0]).reshape(-1,1)
        
        input_y_train_CR = np.concatenate([mc_cr_label, data_cr_label], axis=0)
        
        with open(args.config, 'r') as stream:
            params = yaml.safe_load(stream)
            
        # Define the network
        NN_reweight = Classifier(n_inputs=7, layers=params["layers"], learning_rate=params["learning_rate"], device=device)
        
        print("Training Reweight model... for s/b =",p)
        if not Testing:
            NN_reweight.train(input_x_train_CR, input_y_train_CR, save_model=True, batch_size=params["batch_size"], n_epochs=params["n_epochs"], model_name=f"reweight_{p}", outdir=model_dir)
        
        print("Done training for s/b =", p)
        
    print("Train Reweight Done for seed =", args.gen_seed)

if __name__ == "__main__":
    main()
