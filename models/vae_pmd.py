import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, num_in, model_params_dict):
        super(VAE, self).__init__()
        self.m = model_params_dict.miss_mask_training
        self.input_dim_wm = num_in + self.m*num_in
        self.h_dim_wm = model_params_dict.h_dim + self.m*num_in
        self.device = model_params_dict.device
        #self.device = torch.device("cuda")
        self.num_samples = model_params_dict.num_samples

        self.fc1 = nn.Linear(self.input_dim_wm, model_params_dict.h_dim)
        self.fc2 = nn.Linear(model_params_dict.h_dim, model_params_dict.h_dim)
        self.fc21 = nn.Linear(model_params_dict.h_dim, model_params_dict.z_dim)
        self.fc22 = nn.Linear(model_params_dict.h_dim, model_params_dict.z_dim)
        self.fc3 = nn.Linear(model_params_dict.z_dim, model_params_dict.h_dim)
        self.fc3b = nn.Linear(self.h_dim_wm, model_params_dict.h_dim)
        self.fc4 = nn.Linear(model_params_dict.h_dim, num_in)
        self.mnist = model_params_dict.mnist

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar, test_mode=False, L=1):
        std = torch.exp(0.5*logvar)
        
        if test_mode:
          eps = torch.randn((L, std.shape[0], std.shape[1]))
        else:
          eps = torch.randn_like(std).to(self.device)
        mu = mu.to(self.device)
        eps = eps.to(self.device)
        std = std.to(self.device)
        z = mu + eps*std

        if test_mode:
          z = z.reshape((L*mu.shape[0], mu.shape[1]))

        return z

    def decode(self, z, m):
        h3 = F.relu(self.fc3(z))
        if m.shape[0] != h3.shape[0]:
          m = m.repeat(h3.shape[0],1)
          #print(m)
        h3m = torch.cat((h3,m),1)
        h3b = F.relu(self.fc3b(h3m))
        h4 = self.fc4(h3b)
        if self.mnist:
          h4 = torch.sigmoid(h4)
        return h4

    def forward(self, x, m, test_mode=False, L=1):
      #print("x")
      #print(x) # 0.00 is non-observed, non-zero means observed
      #print("mask")
      #print(m) # 1 or True is non-observed, 0 or False means observed, torch converts True to 1, False to 0
      if self.mnist:
        x = x.view(-1, 784) 
      xm = torch.cat((x,m),1) #batch_size,30*2
      #print(x.shape) #batch_size,30
      mu, logvar = self.encode(xm)
      z = self.reparameterize(mu, logvar, test_mode, L)
      recon = {'xobs': self.decode(z,m), 'xmis': None, 'M_sim_miss': None}
      variational_params = {
        'z_mu': mu, 
        'z_logvar': logvar, 
        'z_mu_prior': torch.zeros_like(mu).to(self.device), 
        'z_logvar_prior': torch.zeros_like(logvar).to(self.device), 
        'qy': None,
        'xmis':  None,
        'xmis_mu': None,
        'xmis_logvar': None,
        'xmis_mu_prior': None,
        'xmis_logvar_prior': None,
      }
      latent_samples = {'z': z}
      return recon, variational_params, latent_samples

    def query_single_attribute(self, x, m, query_attr_index, L=100, test_mode=True):
      print("input")
      print(x)
      xm = torch.cat((x,m),1)
      mu, logvar = self.encode(xm)
      z = self.reparameterize(mu, logvar, test_mode, L)
      recon = self.decode(z,m)
      #print("output")
      #print(recon)
      return recon
