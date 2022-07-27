"""
Used to infer the trained model
"""
import torch
import matplotlib.pyplot as plt

from dataloader import load_nar
from model import Discriminator, FactorVAEGenetic

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} as the accelerator:)")

def load(filepath):
    with open(filepath, 'rb+') as f:
        states = torch.load(f)
    D = Discriminator(z_dim=10).to(device)
    VAE = FactorVAEGenetic().to(device)
    D.load_state_dict(states["model_states"]["D"])
    VAE.load_state_dict(states["model_states"]["VAE"])

    return D, VAE

filepath = "./checkpoints/nar_fvae.pth.tar"
D, VAE = load(filepath)

train_dataloader, _ = load_nar(batch_size=1)
for i, (image, options, solutions) in enumerate(train_dataloader):
    image = image.to(device)
    recons = VAE(image)
    break

image = image.reshape(64, 64).cpu().detach().numpy()
plt.imshow(image)
plt.savefig("./plots/orig.png", dpi=1000)

iter = 1
fig = plt.figure()
columns = 10
rows = 10

for recon_images_in_dim in recons:
    for recon_image in recon_images_in_dim:
        recon_image = recon_image.reshape(64, 64).cpu().detach().numpy()
        fig.add_subplot(rows, columns, iter)
        plt.axis("off")
        plt.imshow(recon_image, aspect='equal')
        iter += 1

plt.savefig(f"./plots/orig_recon_zall.png", dpi=1000)
