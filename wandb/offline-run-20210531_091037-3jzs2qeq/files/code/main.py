import utils.params as params
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys
import torch
import torch.utils.data
import wandb
import math
os.environ['WANDB_NOTEBOOK_NAME'] = 'some text here'
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_MODE'] = 'offline'

from utils.utils import make_deterministic, save_image_reconstructions
from utils.utils import rounding, renormalization, normalization, rmse_loss

from train import train_VAE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

model_map = {
    'mean': 'mean',
    'VAE': train_VAE,
    'VAE_PMD': train_VAE  #Collier
}

config_name = 'binaryAB'
args = params.Params('./config/' + str(config_name)+ '.json')
args.cuda = torch.cuda.is_available()
args.mul_imp = False
args.targets_file = None
args.post_sample = False
if args.mul_imp:
    args.post_sample = True

if args.cuda:
    args.log_interval = 1

args.downstream_logreg = False
args.h_dim = 128 #dimension of hidden layers
args.z_dim = 20 #dimension of latent space
args.weight_decay = 0
args.max_epochs = 500
args.z_beta = 1
args.r_beta = 1
args.xmis_beta = 1
args.learning_rate = 1e-3
args.batch_size = 200
args.miss_mask_training = False
args.r_cat_dim = 10
#args.patience = 50

if config_name == 'binaryAB':
    args.h_dim = 8
    args.z_dim = 4
    args.batch_size = 64
    args.max_epochs = 100
    args.learning_rate = 1e-2
    print("Modified args")
    args.cuda = False
    args.device = torch.device("cpu")

def main(args): 
    for seed in [0,1,2,3,4]:#0 to 4
        for data_file in ['binaryAB']:#'breast','credit'
            for miss_type in ['MNAR1var']: # 'MNAR1var'
                for miss_ratio in [0.8]: #0.8
                    for model_class in ['VAE_PMD']: #, 'VAE_PMD','missForest','mean', 'mice'
                        VAE_model = -1
                        args.seed = seed
                        args.data_file = os.path.join("data",data_file)
                        args.miss_type = miss_type
                        args.miss_ratio = miss_ratio
                        miss_file_name = miss_type + "_uniform_frac_" + str(int(miss_ratio*100)) + "_seed_" + str(seed) 
                        args.compl_data_file = os.path.join(args.data_file,"data_"+str(args.seed))
                        args.miss_data_file = os.path.join(args.data_file,"miss_data",miss_file_name)
                        args.miss_type = miss_type
                        args.miss_ratio = miss_ratio
                        args.model_class = model_class
                        if ('VAE_PMD' == args.model_class):
                            args.miss_mask_training = True
                        args.mnist = ('MNIST' in args.compl_data_file)
                        args.wandb_run_name = model_class+"_"+data_file+"_"+miss_type+"_uniform_frac_"+str(miss_ratio)+"_seed_"+str(seed)
                        args.wandb_tag = 'run_new'
                        args.num_samples = 1
                        # load data 
                        if args.targets_file is not None:
                            files = [args.compl_data_file, args.miss_data_file, args.targets_file]
                        else:
                            files = [args.compl_data_file, args.miss_data_file]
                        train_files, val_files, test_files = [[file + ending for file in files] for ending in ['.train', '.val', '.test']]

                        compl_data_train_ori = pd.read_csv(train_files[0], header=None, dtype=float)
                        compl_data_val_ori = pd.read_csv(val_files[0], header=None, dtype=float)
                        compl_data_test_ori = pd.read_csv(test_files[0], header=None, dtype=float)
                        M_sim_miss_train = pd.DataFrame(pd.read_csv(train_files[1], header=None, dtype=float).values & ~np.isnan(compl_data_train_ori), dtype=bool)
                        M_sim_miss_val = pd.DataFrame(pd.read_csv(val_files[1], header=None, dtype=float).values & ~np.isnan(compl_data_val_ori), dtype=bool)
                        M_sim_miss_test = pd.DataFrame(pd.read_csv(test_files[1], header=None, dtype=float).values & ~np.isnan(compl_data_test_ori), dtype=bool)
                        data_train_ori = compl_data_train_ori.mask(M_sim_miss_train)
                        data_val_ori = compl_data_val_ori.mask(M_sim_miss_val)
                        data_test_ori = compl_data_test_ori.mask(M_sim_miss_test)
                        try: #mnist
                            targets_train = np.squeeze(pd.read_csv(train_files[2], header=None).values) 
                            targets_val = np.squeeze(pd.read_csv(val_files[2], header=None).values) 
                            targets_test = np.squeeze(pd.read_csv(test_files[2], header=None).values) 
                            target_unq = np.unique(targets_train)
                            
                            ohe_targets_train = np.zeros((len(targets_train), len(target_unq)))
                            ohe_targets_val = np.zeros((len(targets_val), len(target_unq)))
                            ohe_targets_test = np.zeros((len(targets_test), len(target_unq)))
                            for targeti in range(len(target_unq)):
                                ohe_targets_train[targets_train==target_unq[targeti], targeti] = 1
                                ohe_targets_val[targets_val==target_unq[targeti], targeti] = 1
                                ohe_targets_test[targets_test==target_unq[targeti], targeti] = 1
                        except IndexError:
                            targets_train, targets_val, targets_test = None, None, None
                        M_sim_miss_train = M_sim_miss_train.values
                        M_sim_miss_val = M_sim_miss_val.values
                        M_sim_miss_test = M_sim_miss_test.values

                        # normalize data 
                        norm_type = 'minmax' * args.mnist + 'standard' * (1-args.mnist)
                        #print(M_sim_miss_train) #False for observed, True for non-observed
                        #print("Data Train Original")
                        #print(data_train_ori.values) #has actual value for observed, has nan for non-observed
                        data_train, norm_parameters = normalization(data_train_ori.values, None, norm_type)
                        data_val, _ = normalization(data_val_ori.values, norm_parameters, norm_type)
                        data_test, _ = normalization(data_test_ori.values, norm_parameters, norm_type)

                        compl_data_train, _ = normalization(compl_data_train_ori.values, norm_parameters, norm_type)
                        compl_data_val, _ = normalization(compl_data_val_ori.values, norm_parameters, norm_type)
                        compl_data_test, _ = normalization(compl_data_test_ori.values, norm_parameters, norm_type)

                        # logging
                        make_deterministic(args.seed)
                        wandb.init(project="miss-vae", name=args.wandb_run_name, tags=[args.wandb_tag], save_code=True)
                        wandb.config.update(args)

                        # compute imputations
                        if (args.model_class=='mice') or (args.model_class=='missForest'):
                            train_imputed = []
                            test_imputed = []
                            for l in range(args.num_samples):
                                if args.model_class == 'mice':
                                    imputer = model_map[args.model_class](random_state=args.seed*l+l, max_iter=10, estimator=BayesianRidge(), sample_posterior=args.post_sample) 
                                elif args.model_class == 'missForest':
                                    imputer = model_map[args.model_class](random_state=args.seed*l+l, max_iter=10, estimator=ExtraTreesRegressor(n_estimators=10, n_jobs=2)) 
                                train_imputed.append(imputer.fit_transform(data_train))
                                test_imputed.append(imputer.transform(data_test))
                            if not args.mul_imp:
                                train_imputed = np.mean(train_imputed, axis=0) 
                                test_imputed = np.mean(test_imputed, axis=0)
                        else:
                            if args.model_class == 'mean':
                                data_train_df = pd.DataFrame(data_train)
                                data_test_df = pd.DataFrame(data_test)
                                train_imputed = data_train_df.fillna(data_train_df.mean(), inplace=False).values
                                test_imputed = data_test_df.fillna(data_train_df.mean(), inplace=False).values         
                            if 'VAE' in args.model_class:
                                VAE_model,train_imputed, train_imputed_1, test_imputed = model_map[args.model_class](data_train, data_test, compl_data_train, compl_data_test, wandb, args, norm_parameters)
                            elif (args.model_class == 'miwae') or (args.model_class == 'notmiwae'):
                                train_imputed, test_imputed = model_map[args.model_class](compl_data_train, data_train, compl_data_test, compl_data_test, norm_parameters, wandb, args)
                    
                        if args.mnist:
                            save_image_reconstructions(train_imputed*M_sim_miss_train+compl_data_train*(1-M_sim_miss_train), compl_data_train, M_sim_miss_train, 28, 'images', args.wandb_run_name)
                        
                        # compute losses
                        M_obs_train, M_obs_test = ~M_sim_miss_train & ~np.isnan(compl_data_train), ~M_sim_miss_test & ~np.isnan(compl_data_test)

                        if not args.mul_imp:
                            # renormalization
                            train_imputed = renormalization(train_imputed, norm_parameters, norm_type)
                            test_imputed = renormalization(test_imputed, norm_parameters, norm_type)
                            compl_data_train = renormalization(compl_data_train, norm_parameters, norm_type)
                            compl_data_test = renormalization(compl_data_test, norm_parameters, norm_type)

                            # rounding
                            train_imputed = rounding(train_imputed, compl_data_train)
                            test_imputed = rounding(test_imputed, compl_data_test)
                            test_imputed_df = pd.DataFrame(test_imputed)
                            test_imputed_df.to_csv(os.path.join(args.data_file, "miss_data", miss_file_name + "imputed_test.csv"), header=False, index=False)
                        else:
                            # renormalization
                            train_imputed = [renormalization(train_imputed[i], norm_parameters, norm_type) for i in range(len(train_imputed))]
                            test_imputed = [renormalization(test_imputed[i], norm_parameters, norm_type) for i in range(len(test_imputed))]
                            compl_data_train = renormalization(compl_data_train, norm_parameters, norm_type)
                            compl_data_test = renormalization(compl_data_test, norm_parameters, norm_type)

                            # rounding
                            train_imputed = [rounding(train_imputed[i], compl_data_train) for i in range(len(train_imputed))]
                            test_imputed = [rounding(test_imputed[i], compl_data_test) for i in range(len(test_imputed))]
                        # save imputations
                        if not args.mul_imp:
                            try:
                                imputed_dir = '/'.join(args.compl_data_file.split('/')[:-1] + ['imputed'])
                                from pathlib import Path
                                Path(imputed_dir).mkdir(parents=True, exist_ok=True)    
                            except FileExistsError:
                                pass
                            np.savetxt(imputed_dir + f'/{args.wandb_run_name}.train', train_imputed, delimiter=',')
                            np.savetxt(imputed_dir + f'/{args.model_class}.test', test_imputed, delimiter=',')
                        
                            # compute loss
                            if args.model_class == 'hivae':
                                train_mis_mse = rmse_loss(train_imputed, compl_data_train[:len(train_imputed)], M_sim_miss_train[:len(train_imputed)])
                                test_mis_mse = rmse_loss(test_imputed, compl_data_test[:len(test_imputed)], M_sim_miss_test[:len(test_imputed)])
                            else:
                                train_mis_mse = rmse_loss(train_imputed, compl_data_train, M_sim_miss_train)
                                test_mis_mse = rmse_loss(test_imputed, compl_data_test, M_sim_miss_test)

                            # log loss
                            wandb.log({'Train Imputation RMSE loss': train_mis_mse})
                            wandb.log({'Test Imputation RMSE loss': test_mis_mse})

                            # loss for a single importance sample
                            if 'VAE' in args.model_class:
                                train_imputed_1 = renormalization(train_imputed_1, norm_parameters, norm_type)
                                train_imputed_1 = rounding(train_imputed_1, compl_data_train)
                                train_1_mis_mse = rmse_loss(train_imputed_1, compl_data_train, M_sim_miss_train)
                                wandb.log({'Train Imputation RMSE loss (single sample)': train_1_mis_mse})

                            with open('table.txt', "a") as myfile:
                                myfile.write(','.join(map(str, [args.miss_data_file, args.seed, args.model_class, train_mis_mse, test_mis_mse])) + '\n')

                            print(','.join(map(str, [args.miss_data_file, args.seed, args.model_class, train_mis_mse, test_mis_mse])))

                        else:
                            data_x_train = np.concatenate(train_imputed, 0)
                            data_x_test = np.concatenate(test_imputed, 0)
                            if args.downstream_logreg:
                                targets_train_full = np.tile(ohe_targets_train, (args.num_samples,1))
                                targets_test_full = np.tile(ohe_targets_test, (args.num_samples,1))
                                clf = LogisticRegression(random_state=args.seed).fit(data_x_train, targets_train_full.argmax(1))
                                test_acc = clf.score(data_x_test, targets_test_full.argmax(1))

                                wandb.log({'Test accuracy': test_acc})
                                print(test_acc)
                        return VAE_model, norm_parameters, norm_type

if __name__ == "__main__":
    #queryBgivenA = np.array([[0,math.nan],[1,math.nan]]) #A is 1
    #queryBgivenA[queryBgivenA!=queryBgivenA] = 0 #nan values set to 0
    #print(queryBgivenA)

    VAE_model, norm_parameters, norm_type = main(args)
    print(norm_parameters)
    mask = np.array([[False,True],[False,True]])
    mask = torch.tensor(mask).to(args.device).float()

    queryBgivenA = np.array([[0,math.nan],[1,math.nan]]) #A index = 0
    query_normalized = renormalization(queryBgivenA, norm_parameters, norm_type)
    query_normalized[query_normalized!=query_normalized] = 0 #nan values set to 0
    query_normalized = torch.tensor(query_normalized).to(args.device).float()
    #ToDO write function to create mask based on nan values

    #print(mask)
    #print(query_normalized)
    query_attr_index = 1
    xB_query = VAE_model.query_single_attribute(query_normalized[0].unsqueeze(0), mask[0].unsqueeze(0), query_attr_index, L = 100)
    xB_query = renormalization(xB_query.cpu().detach().numpy(), norm_parameters, norm_type)
    #print(xB_query)
    xB_query = rounding(xB_query,np.ones(xB_query.shape)) #since all categorical
    unique, counts = np.unique(xB_query[:,query_attr_index], return_counts=True)
    xB_query = dict(zip(unique, counts))
    print("P(B|A=0)")
    print(xB_query)
    print("All done!")

    xB_query = VAE_model.query_single_attribute(query_normalized[1].unsqueeze(0), mask[1].unsqueeze(0), query_attr_index, L = 100)
    xB_query = renormalization(xB_query.cpu().detach().numpy(), norm_parameters, norm_type)
    #print(xB_query)
    xB_query = rounding(xB_query,np.zeros(xB_query.shape)) #since all categorical
    unique, counts = np.unique(xB_query[:,query_attr_index], return_counts=True)
    xB_query = dict(zip(unique, counts))
    print("P(B|A=1)")
    print(xB_query)
    print("All done!")