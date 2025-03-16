import argparse
import numpy as np
import torch
import os
import logging
import yaml
import sys
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal

script_dir = os.path.dirname(__file__) 
helpers_path = os.path.join(script_dir, '..', 'model_scripts')  
sys.path.insert(0, os.path.abspath(helpers_path))
from SimpleMAF import SimpleMAF

parser = argparse.ArgumentParser()
parser.add_argument("-i","--indir",help="working folder",
    default="/home/aegis/Titan0/NRAD/SPP_NRAD"
)
parser.add_argument("-s","--signal",default=None,help="signal fraction",)
parser.add_argument("-c","--config",help="Morph flows config file",default="configs/morph_physics.yml")
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument("-v","--verbose",default=False,help="Verbose enable DEBUG",)
#parser.add_argument("-cu", "--cuda_slot", help = "cuda_slot")

args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)

logging.basicConfig(level=logging.INFO)
log_level = logging.DEBUG if args.verbose else logging.INFO
log = logging.getLogger("run")
log.setLevel(log_level)


def main():
    Testing = False
    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    #device = torch.device("cpu")
    
    seed_path = f"{args.indir}/data/seed{args.gen_seed}/"
    model_dir = f"{args.indir}/models/seed{args.gen_seed}/"
    os.makedirs(model_dir, exist_ok=True)
        
    mc_path = f"{args.indir}/data/mc_events.npz"
    mc_events = np.load(mc_path)
    mc_events_cr = mc_events["mc_events_cr"]
    mc_events_sr = mc_events["mc_events_sr"]
    
    per = []
    if args.signal is not None:
        per = [args.signal]
    else:
        per = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024]
    
    for p in per:
        print("Working with s/b =", p)
        
        # load input files
        data_events = np.load(f"{seed_path}/data_{p}.npz")
        data_events_cr = data_events["data_events_cr"]
    
        print("CR has", len(data_events_cr), "data events,", len(mc_events_cr), "MC events.")

        # Train flow in the CR
        # To do the closure tests, we need to withhold a small amount of CR data
        n_withold = 10000 
        n_context = 2
        n_features = 5
    
        data_context_cr_train = data_events_cr[:-n_withold,:n_context]
        data_context_cr_test = data_events_cr[-n_withold:,:n_context]
        data_feature_cr_train = data_events_cr[:-n_withold,n_context:]
        data_feature_cr_test = data_events_cr[-n_withold:,n_context:]
    
        mc_context_cr_train = mc_events_cr[:-n_withold,:n_context]
        mc_context_cr_test = mc_events_cr[-n_withold:,:n_context]
        mc_feature_cr_train = mc_events_cr[:-n_withold,n_context:]
        mc_feature_cr_test = mc_events_cr[-n_withold:,n_context:]
    
        mc_context_sr = mc_events_sr[:,:n_context]
        mc_feature_sr = mc_events_sr[:,n_context:]
    
        with open(args.config, 'r') as stream:
            params = yaml.safe_load(stream)
     
        # Define the base density flow
        base_density_flow = SimpleMAF(num_features=n_features, num_context=n_context, device=device, num_layers=params["base"]["n_layers"], num_hidden_features=params["base"]["n_hidden_features"], learning_rate=params["base"]["learning_rate"])
    
        print("Training Morph model (base)...")
        if not Testing:
            base_density_flow.train(data=mc_feature_cr_train, cond=mc_context_cr_train, batch_size=params["base"]["batch_size"], n_epochs=params["base"]["n_epochs"], outdir=model_dir, save_model=True, model_name=f"morph_base_{p}")
        
        print("Done training base!")
        
        # Define the top transformer flow
        transport_flow = SimpleMAF(num_features = n_features, num_context=n_context, base_dist=base_density_flow.flow, num_layers=params["top"]["n_layers"], num_hidden_features=params["top"]["n_hidden_features"], learning_rate=params["top"]["learning_rate"], device=device)
        
        print("Training Morph model (top)...")
        if not Testing:
            transport_flow.train(data=data_feature_cr_train, cond=data_context_cr_train, batch_size=params["top"]["batch_size"], n_epochs=params["top"]["n_epochs"], outdir=model_dir, save_model=True, model_name=f"morph_top_{p}")
        print("Done training top!")
        print("Done training for s/b =", p)
        
      
    print("Train Morph Done for seed", args.gen_seed)

    
    
if __name__ == "__main__":
    main()
