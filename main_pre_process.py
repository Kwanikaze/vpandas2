import utils.params as params
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import preprocess as data
args = params.Params('./config/binaryAB.json')
args.uniform=1
args.header = None
args.train_pct = 0.8
args.val_pct = 0.1
args.n = None
args.m = None

miss_type_map = {
    'MCAR': data.MCAR,
    'MAR': data.MAR,
    'label': data.label_dependent_missingness_no_noise,
    'labelnoise': data.label_dependent_missingness,
    'logit': data.logit_missingness,
    'MNARsum': data.MNARsum,
    'MNAR1var': data.MNAR1var,
    'MNAR1varMCAR': data.MNAR1varMCAR,
    'MNAR2var': data.MNAR2var,
}

for seed in [0,1,2,3,4]:
    for data_file in ['breast']:
        for miss_type in ['MCAR']:
            for miss_ratio in [0.20, 0.80]:
                args.seed = seed
                args.data_file = os.path.join("data",data_file)
                args.miss_type = miss_type
                args.miss_ratio = miss_ratio

                miss_file_name = args.miss_type + "_" + (1-args.uniform)*"not" + "uniform_frac_" + str(int(args.miss_ratio*100)) + "_seed_" + str(args.seed)  
                Path(os.path.join(args.data_file, "miss_data")).mkdir(parents=True, exist_ok=True)
                print("File " + args.data_file + "/miss_data/" + miss_file_name + " will now be generated.")

                # load data
                data = pd.read_csv(os.path.join(args.data_file, "data.csv"), engine="python", header=args.header)
                try: 
                    targets = pd.read_csv(os.path.join(args.data_file, "targets.csv"), header=None)
                except FileNotFoundError:
                    pass

                # robustness study
                if args.n is not None:
                    miss_file_name += '_n_' + str(args.n)
                    data = data.sample(args.n, axis=0)
                if args.m is not None:
                    miss_file_name += '_m_' + str(args.m)
                    data = data.sample(args.m, axis=1)

                # induce missingness
                np.random.seed(args.seed)
                induce_missingness = miss_type_map[args.miss_type]
                if 'label' in args.miss_type:
                    labels = pd.read_csv(os.path.join(args.data_file, "targets.csv"), engine="python", header=None)
                    M, patternsets = induce_missingness(data, labels, missingness_ratio=args.miss_ratio, seed=args.seed)
                else:
                    M, patternsets = induce_missingness(data, missingness_ratio=args.miss_ratio, seed=args.seed)

                # train test split
                train_idx = int(args.train_pct*len(data))
                val_idx = int((args.train_pct+args.val_pct)*len(data))
                random_permute = np.random.RandomState(seed=args.seed).permutation(len(data))
                M_train, M_val, M_test = M.iloc[random_permute[:train_idx]], M.iloc[random_permute[train_idx:val_idx]], M.iloc[random_permute[val_idx:]]  
                data_train, data_val, data_test = data.iloc[random_permute[:train_idx]], data.iloc[random_permute[train_idx:val_idx]], data.iloc[random_permute[val_idx:]] 

                 # save induced missing data
                M_train.to_csv(os.path.join(args.data_file, "miss_data", miss_file_name + ".train"), header=False, index=False)
                M_val.to_csv(os.path.join(args.data_file, "miss_data", miss_file_name + ".val"), header=False, index=False)
                M_test.to_csv(os.path.join(args.data_file, "miss_data", miss_file_name + ".test"), header=False, index=False) 

                data_name = f"data_{args.seed}"
                if args.n is not None:
                    data_name += f"_n_{args.n}"
                if args.m is not None:
                    data_name += f"_m_{args.m}"
                data_train.to_csv(os.path.join(args.data_file, f"{data_name}.train"), header=False, index=False)
                data_val.to_csv(os.path.join(args.data_file, f"{data_name}.val"), header=False, index=False)
                data_test.to_csv(os.path.join(args.data_file, f"{data_name}.test"), header=False, index=False)

                if patternsets is not None:
                    patternsets_train, patternsets_val, patternsets_test = patternsets.iloc[random_permute[:train_idx]], patternsets.iloc[random_permute[train_idx:val_idx]], patternsets.iloc[random_permute[val_idx:]]  
                    patternsets_train.to_csv(os.path.join(args.data_file, "miss_data", f"{miss_file_name}_patternsets.train"), header=False, index=False)
                    patternsets_val.to_csv(os.path.join(args.data_file, "miss_data", f"{miss_file_name}_patternsets.val"), header=False, index=False)
                    patternsets_test.to_csv(os.path.join(args.data_file, "miss_data", f"{miss_file_name}_patternsets.test"), header=False, index=False)
                
                    try:
                        targets_train, targets_val, targets_test = targets.iloc[random_permute[:train_idx]], targets.iloc[random_permute[train_idx:val_idx]], targets.iloc[random_permute[val_idx:]]  
                        targets_train.to_csv(os.path.join(args.data_file, f"targets_{args.seed}.train"), header=False, index=False)
                        targets_val.to_csv(os.path.join(args.data_file, f"targets_{args.seed}.val"), header=False, index=False)
                        targets_test.to_csv(os.path.join(args.data_file, f"targets_{args.seed}.test"), header=False, index=False)
                    except UnboundLocalError:
                        pass