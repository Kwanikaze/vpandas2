import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def read_csv(data_dir):
  df = pd.read_csv(data_dir)
  df.dropna() #drop rows with at least one element missing
  return df

def label_encode_columns(df,columns):
  labelencoder = LabelEncoder()
  for col in columns:
    df[col] = labelencoder.fit_transform(df[col])
  return df

def one_hot_encode_columns(df,columns_to_OHE,args):
  df[columns_to_OHE]= df[columns_to_OHE].astype(int)
  for col in columns_to_OHE:
    col_OHE = pd.get_dummies(prefix = col,data= df[col])
    #Generate Unif[0,1] noise with the same dimension as col_OHE
    if args.add_noise == "True":
      noise = np.random.uniform(low=0.0,high=1.0,size=col_OHE.shape)
      col_OHE = col_OHE + noise
    df = df.join(col_OHE)
  df.drop(columns_to_OHE,axis=1,inplace=True)
  return df

def unif_noise_to_real_columns(df,real_vars):
  for col in real_vars:
    noise = np.random.uniform(low=0.0,high=0.01,size=df[col].shape)
    df[col] = df[col]+noise
  return df

def duplicate_dataframe(df,duplications):
  df = pd.DataFrame(np.tile(df, (duplications, 1)),columns = df.columns)
  return df


def standarize_real_columns(df,real_vars): #Standardize to 0,1
  mms_dict = {}
  for col in real_vars:
    mms_dict[col] = MinMaxScaler()
    df[col] = mms_dict[col].fit_transform(df[col].values.reshape(-1, 1)) #.values returns a np array
  return df, mms_dict


def preprocess(df_raw,args, real_vars, cat_vars, duplications=100):
  df = duplicate_dataframe(df_raw, duplications)
  df = df.sample(frac=1, random_state=args.random_seed)
  #print(df)
  if args.add_noise == "True":
    df = unif_noise_to_real_columns(df, real_vars)
  df, min_max_scalar_dict = standarize_real_columns(df,real_vars)
  df_OHE = one_hot_encode_columns(df, cat_vars, args)
  #print(df_OHE)
  return df, df_OHE, min_max_scalar_dict

def split(df,df_OHE,split_pct):
  train_df, val_df, test_df = np.split(df, [int(split_pct[0]*len(df)), int(split_pct[1]*len(df))])
  train_df_OHE, val_df_OHE, test_df_OHE = np.split(df_OHE, [int(split_pct[0]*len(df_OHE)), int(split_pct[1]*len(df_OHE))])
  print(train_df_OHE.shape,val_df_OHE.shape,test_df_OHE.shape)
  return train_df, train_df_OHE, val_df, val_df_OHE, test_df, test_df_OHE

