import pickle
import matplotlib.pyplot as plt

with open("./losses", "rb") as f:
    losses = pickle.load(f)["losses"]

vae_recon_loss, vae_kld_loss, vae_tc_loss = [], [], []
for loss in losses:
    vae_recon_loss.append(loss[0])
    vae_kld_loss.append(loss[1])
    vae_tc_loss.append(loss[2])

def line_plot(loss, title, filename):
    plt.plot(loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(filename, dpi=600)
    plt.show()

line_plot(vae_recon_loss, "VAE Reconstruction Loss vs Epochs", "./plots/recon_loss")
line_plot(vae_kld_loss, "VAE KLD Loss vs Epochs", "./plots/kld_loss")
line_plot(vae_tc_loss, "VAE Total Correlation Loss vs Epochs", "./plots/tc_loss")
