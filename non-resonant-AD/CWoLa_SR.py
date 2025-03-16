import argparse
import numpy as np
from model_scripts.Classifier import *
import torch
import os
import sys
from sklearn.metrics import roc_auc_score
import argparse


# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
#parser.add_argument("-cu", "--cuda_slot", help = "cuda_slot")
parser.add_argument("-n", "--classifier_runs", help = "classifier_runs", default=20)
parser.add_argument("-i","--indir",help="home folder",default="/home/aegis/Titan0/NRAD/SPP_NRAD")
parser.add_argument("-s","--signal",default=None,help="signal fraction")
parser.add_argument("-c","--config",help="BC config file",default="configs/bc_discrim.yml")
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument("-full_sup",action='store_true',help="Run fully supervised case")
parser.add_argument("-ideal",action='store_true',help="Run idealized classifier")
parser.add_argument("-reweight",action='store_true',help="Run Reweight method")
parser.add_argument("-generate",action='store_true',help="Run Generate method")
parser.add_argument("-morph",action='store_true',help="Run Morph method")

# Read arguments from command line
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)

def regularize_weights(w_arr, sigma = 3.0):
    w_copy = np.copy(w_arr)
    mean_w = np.mean(w_copy)
    std_w = np.std(w_copy)
    w_copy[w_copy > (sigma*std_w + mean_w)] = 0
    return w_copy
    

def run_eval(set_1, set_2, code, save_dir, classifier_params, device, w_1 = None, w_2 = None, run_test = False, test_B = None, test_S = None, crop_weights = True):
    
    if w_1 is None:
        w_1 = np.array([1.]*set_1.shape[0])
    if w_2 is None:
        w_2 = np.array([1.]*set_2.shape[0])
    if crop_weights:
        w_1 = regularize_weights(w_1)
        w_2 = regularize_weights(w_2)
    
    input_x_train = np.concatenate([set_1, set_2], axis=0)
    input_y_train = np.concatenate([np.zeros(set_1.shape[0]).reshape(-1,1), np.ones(set_2.shape[0]).reshape(-1,1)], axis=0)
    input_w_train = np.concatenate([w_1, w_2], axis=0).reshape(-1, 1)
    
    print(f"Working on {code}...")
    print("      X train, y train, w train:", input_x_train.shape, input_y_train.shape, input_w_train.shape)
    
    if run_test:
        input_x_test = np.concatenate([test_B, test_S], axis=0)
        input_y_test = np.concatenate([np.zeros(test_B.shape[0]).reshape(-1,1), np.ones(test_S.shape[0]).reshape(-1,1)], axis=0)
        print("      X test, y test:", input_x_test.shape, input_y_test.shape)

    for i in range(int(args.classifier_runs)):
        
        print(f"Classifier run {i+1} of {args.classifier_runs}.")
        local_id = f"{code}_run{i}"
                
        # train classifier
        NN = Classifier(n_inputs=5, layers=classifier_params["layers"], learning_rate=classifier_params["learning_rate"], device=device, scale_data=False)
        NN.train(input_x_train, input_y_train, weights=input_w_train,  save_model=True, model_name = f"model_{local_id}" , n_epochs=classifier_params["n_epochs"], seed = i, outdir=save_dir)

        if run_test:
            scores = NN.evaluation(input_x_test)
            auc = roc_auc_score(input_y_test, scores)
            if auc < 0.5: auc = 1.0 - auc
            print(f"   AUC: {auc}")
            os.makedirs(f"{save_dir}/auc_scores", exist_ok=True)
            np.savez(f"{save_dir}/auc_scores/auc_{code}.npz", auc_scores = auc)
                
    print()


def main():
    
    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    static_data_dir = f"{args.indir}/data/"
    seeded_data_dir = f"{args.indir}/data/seed{args.gen_seed}/"
    samples_dir = f"{args.indir}/samples/seed{args.gen_seed}/"
    eval_dir = f"{args.indir}/evaluation/seed{args.gen_seed}/"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load in the classifier params
    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
        
    n_context = 2

 
    # load in the test sets
    test_events = np.load(f"{static_data_dir}/test_SR.npz")
    test_bkg = test_events["bkg_events_SR"][:,n_context:]
    test_sig = test_events["sig_events_SR"][:,n_context:]
    
    path_to_data = f"{seeded_data_dir}/data_{args.signal}.npz"
        
#     if args.full_sup:
#         full_sup_events = np.load(f"{static_data_dir}/fullsup_SR.npz")
#         set_1 = full_sup_events["bkg_events_SR"][:,n_context:]
#         set_2 = full_sup_events["sig_events_SR"][:,n_context:]
#         run_eval(set_1, set_2, code="full_sup", save_dir=eval_dir, classifier_params=params, device=device, run_test=True, test_B=test_bkg, test_S=test_sig)
#         print()
        
    if args.signal is not None:
        per = [args.signal]
    else:
        per = [0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024]
    
    for p in per:
        if args.reweight:
            reweight_events = np.load(f"{samples_dir}/reweight_SR_samples_{p}.npz")
            data_events = np.load(path_to_data)    
            set_1 = reweight_events["mc_samples"][:,n_context:]
            w_1 =  reweight_events["w_sr"]
            set_2 = data_events["data_events_sr"][:,n_context:]
            run_eval(set_1, set_2, w_1 = w_1, code=f"reweight_{p}_sr", save_dir=eval_dir, classifier_params=params, device=device, run_test=True, test_B=test_bkg, test_S=test_sig, crop_weights=True)
            print()
            
        if args.generate:
            generate_events = np.load(f"{samples_dir}/generate_SR_samples_{p}.npz")
            context_weights = np.load(f"{samples_dir}/context_weight_SR_samples_{p}.npz")
            data_events = np.load(path_to_data)    
            set_1 = generate_events["samples"]
            w_1 =  context_weights["w_sr"]
            set_2 = data_events["data_events_sr"][:,n_context:]
            run_eval(set_1, set_2, w_1 = w_1, code=f"generate_{p}_sr", save_dir=eval_dir, classifier_params=params, device=device, run_test=True, test_B=test_bkg, test_S=test_sig, crop_weights=True)
            print()
            
        if args.morph:
            morph_events = np.load(f"{samples_dir}/morph_SR_samples_{p}.npz")
            context_weights = np.load(f"{samples_dir}/context_weight_SR_samples_{p}.npz")
            data_events = np.load(path_to_data)    
            set_1 = morph_events["samples"]
            w_1 =  context_weights["w_sr"]
            set_2 = data_events["data_events_sr"][:,n_context:]
            run_eval(set_1, set_2, w_1 = w_1, code=f"morph_{p}_sr", save_dir=eval_dir, classifier_params=params, device=device, run_test=True, test_B=test_bkg, test_S=test_sig, crop_weights=True)
            print()
            
    print("All done!")

        
    
if __name__ == "__main__":
    main()
