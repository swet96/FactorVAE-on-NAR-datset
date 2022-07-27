import os
import torch
import pickle
import logging
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from datetime import datetime
from dataloader import load_nar
from model import FactorVAE, Discriminator, FactorVAE2
from utils import recon_loss, kl_divergence, permute_dims

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dataloader, _ = load_nar(num_task=10000)

##### CONSTANTS #####
lr_VAE     = 1e-6
lr_D       = 1e-5
beta1_VAE  = 0.9
beta2_VAE  = 0.999
beta1_D    = 0.5
beta2_D    = 0.9
weight_decay_VAE = 0.15
# weight_decay_D = 0.15
epochs     = 5000
z_dim      = 20
gamma      = 6.4
batch_size = 64
patience   = 10

ckpt_dir = "./checkpoints/"
ckptname="nar_fvae_final.pth.tar"
logger_filename= f"./logger/nar_fvae_final.log"
#####################

# Create and configure logger
logging.basicConfig(filename= logger_filename,
                    format='%(asctime)s %(message)s',
                    filemode='a')
 
# Creating an object
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ones = torch.ones(batch_size, dtype=torch.long, device=device)
zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
pbar = tqdm(total=epochs)


def save_checkpoint(D, VAE, optim_D, optim_VAE, epoch, ckptname, verbose=True):
    model_states = {'D': D.state_dict(),
                    'VAE': VAE.state_dict()}
    optim_states = {'optim_D': optim_D.state_dict(),
                    'optim_VAE': optim_VAE.state_dict()}
    states = {'iter': epoch,
              'model_states': model_states,
              'optim_states': optim_states}

    filepath = os.path.join(ckpt_dir, str(ckptname))
    with open(filepath, 'wb+') as f:
        torch.save(states, f)
    if verbose:
        pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, epoch))

def load(filepath):
    with open(filepath, 'rb+') as f:
        states = torch.load(f)
    D = Discriminator(z_dim=10).to(device)
    VAE = FactorVAE().to(device)
    D.load_state_dict(states["model_states"]["D"])
    VAE.load_state_dict(states["model_states"]["VAE"])
    iter = states["iter"]

    return D, VAE, iter

try:
    D, VAE, iter = load(os.path.join(ckpt_dir, str(ckptname)))
    print("Found checkpoints :)")
except Exception as e:
    print(e)
    VAE = FactorVAE().to(device)
    D   = Discriminator(z_dim=10).to(device)
    iter = 0

# optim_VAE = optim.Adam(VAE.parameters(), lr=lr_VAE,
#                        betas=(beta1_VAE, beta2_VAE), weight_decay=weight_decay_VAE)
optim_VAE = optim.Adam(VAE.parameters(), lr=lr_VAE,
                       betas=(beta1_VAE, beta2_VAE))
optim_D = optim.Adam(D.parameters(), lr=lr_D,
                     betas=(beta1_D, beta2_D))

losses = []

pbar.update(iter)
prev_loss = 10000
epoch_since_last_improve = 0
criterion = nn.CrossEntropyLoss().to(device)
for epoch in range(iter, epochs):
    pbar.update(1)
    # x_true = (batch_size,C,H,W), options =(batch_size, num_options,C,H,W), solution =  (batch_size,C,H,W)
    print(len(train_dataloader))
    for i, (x_true, options, solution) in enumerate(train_dataloader):
        x_true_clone = x_true.clone()

        x_true = x_true.to(device)
        x_true_clone = x_true_clone.to(device)
        x_recon, mu, logvar, z = VAE(x_true)
        vae_recon_loss = recon_loss(x_true, x_recon)
        vae_kld = kl_divergence(mu, logvar)

        D_z = D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        vae_loss = vae_recon_loss + vae_kld + gamma*vae_tc_loss

        optim_VAE.zero_grad()
        vae_loss.backward(retain_graph=True)
        # optim_VAE.step()
        
        z_prime = VAE(x_true_clone, no_dec=True)
        z_pperm = permute_dims(z_prime).detach()
        D_z_pperm = D(z_pperm)
        D_tc_loss = 0.5*(criterion(D_z, zeros) + criterion(D_z_pperm, ones))

        optim_D.zero_grad()
        D_tc_loss.backward()
        optim_VAE.step()
        optim_D.step()
    
    pbar.write('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                    epoch+1, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))
    logger.info('[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                    epoch+1, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))
    
    epoch_since_last_improve += 1
    if (vae_loss + D_tc_loss) < prev_loss:
        prev_loss = vae_loss + D_tc_loss
        pbar.write("Loss reduced!!!!")
        epoch_since_last_improve = 0
        save_checkpoint(D, VAE, optim_D, optim_VAE, epoch, ckptname)

    losses.append([vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item()])
    if epoch_since_last_improve >= patience:
        logger.info("Early stopping hit!!!")
        logger.info(f"Total vae loss kept on increasing for the last {patience} epochs")

        # lr_Dis = float(input("Enter new learning rate for D"))
        # lr_Vae = float(input("Enter new learning rate for VAE"))

        # for g in optim_D.param_groups:
        #     g['lr'] = lr_Dis
        # for g in optim_VAE.param_groups:
        #     g['lr'] = lr_Vae
    with open("./losses", 'wb') as f:
        pickle.dump({"losses": losses}, f)

pbar.write("Training Finished!!!!")
pbar.close()
