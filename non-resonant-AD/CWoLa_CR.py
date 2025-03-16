import argparse
import numpy as np
from model_scripts.Classifier import *
import torch
import os
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

#parser.add_argument("-cu", "--cuda_slot", help = "cuda_slot")
parser.add_argument("-n", "--classifier_runs", help = "classifier_runs",default=20)
parser.add_argument("-i","--indir",help="home folder",default="/home/aegis/Titan0/NRAD/SPP_NRAD")
parser.add_argument("-c","--config",help="BC config file",default="configs/bc_discrim.yml")
parser.add_argument("-g","--gen_seed",help="Random seed for signal injections",default=1)
parser.add_argument("-reweight",action='store_true',help="Run Reweight method")
parser.add_argument("-generate",action='store_true',help="Run Generate method")
parser.add_argument("-morph",action='store_true',help="Run Morph method")

args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda_slot)

def run_eval(set_1, set_2, code, save_dir, classifier_params, device, w_1 = None, w_2 = None):
    #Set 1 Samples
    #Set 2 Data
    #Set the weights to 1 if None
    if w_1 is None or w_1.size == 0:
        w_1 = np.array([1.]*set_1.shape[0])
    if w_2 is None or w_2.size == 0:
        w_2 = np.array([1.]*set_2.shape[0])
        
    # define testsets from the input data
    num_test = 10000

    trainset_1 = set_1[:-num_test]
    trainset_2 = set_2[:-num_test]

    wtrain_1 = w_1[:-num_test]
    wtrain_2 = w_2[:-num_test]

    testset_1 = set_1[-num_test:]
    testset_2 = set_2[-num_test:]
    
    input_x_train = np.concatenate([set_1, set_2], axis=0)
    # 1 for Data and 0 for sample
    input_y_train = np.concatenate([np.zeros(set_1.shape[0]).reshape(-1,1), np.ones(set_2.shape[0]).reshape(-1,1)], axis=0)
    
    input_w_train = np.concatenate([w_1, w_2], axis=0).reshape(-1, 1)

    input_x_test = np.concatenate([testset_1, testset_2], axis=0)
    input_y_test = np.concatenate([np.zeros(testset_1.shape[0]).reshape(-1,1), np.ones(testset_2.shape[0]).reshape(-1,1)], axis=0)
    
    print(f"Working on {code}...")
    print("      X train, y train, w train:", input_x_train.shape, input_y_train.shape, input_w_train.shape)
    print("      X test, y test:", input_x_test.shape, input_y_test.shape)
    
    aucs_list = []

    for i in range(int(args.classifier_runs)):
        
        print(f"Classifier run {i+1} of {args.classifier_runs}.")
        local_id = f"{code}_run{i}"
                
        # train classifier
        NN = Classifier(n_inputs=5, layers=classifier_params["layers"], learning_rate=classifier_params["learning_rate"], device=device, scale_data=False)
        NN.train(input_x_train, input_y_train, weights=input_w_train,  save_model=True, model_name = f"model_{local_id}" , n_epochs=classifier_params["n_epochs"], seed = i, outdir=save_dir)

        scores = NN.evaluation(input_x_test)
        auc = roc_auc_score(input_y_test, scores)
        if auc < 0.5: auc = 1.0 - auc
        aucs_list.append(auc)
        print(f"   AUC: {auc}")
    
    os.makedirs(f"{save_dir}/auc_scores", exist_ok=True)
    np.savez(f"{save_dir}/auc_scores/auc_{code}.npz", auc_scores = auc)
    print("Median auc, 16th percentile, 84th percentile")
    print(np.median(aucs_list), [np.percentile(aucs_list, 16), np.percentile(aucs_list, 84)])
        
    print("Done.")

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
    n_samples = 30000 #ideal bkg and data tend to be rather large
    sig_per = 0
        
    if args.reweight:
        
        reweight_events = np.load(f"{samples_dir}/reweight_CR_samples_{sig_per}.npz")
        set_1 = reweight_events["mc_cr"][:,n_context:]
        w_1 =  reweight_events["w_cr"]
        set_2 = reweight_events["target_cr"][:,n_context:]
        run_eval(set_1, set_2, w_1 = w_1, code=f"reweight_{sig_per}_cr", save_dir=eval_dir, classifier_params=params, device=device)

    if args.generate:
        
        generate_events = np.load(f"{samples_dir}/generate_CR_samples_{sig_per}.npz")
        context_weights = np.load(f"{samples_dir}/context_weight_CR_samples_{sig_per}.npz")
        set_1 = generate_events["generate_cr"]
        set_2 = generate_events["target_cr"]
        run_eval(set_1, set_2, code=f"generate_{sig_per}_cr", save_dir=eval_dir, classifier_params=params, device=device)
        
    if args.morph:

        morph_events = np.load(f"{samples_dir}/morph_CR_samples_{sig_per}.npz")
        context_weights = np.load(f"{samples_dir}/context_weight_CR_samples_{sig_per}.npz")
        set_1 = morph_events["morph_cr"]
        w_1 =  context_weights["w_cr"]
        set_2 = morph_events["target_cr"]
        run_eval(set_1, set_2, w_1 = w_1, code=f"morph_{sig_per}_cr", save_dir=eval_dir, classifier_params=params, device=device)
        
    
    print("All done!")

        
    
if __name__ == "__main__":
    main()
