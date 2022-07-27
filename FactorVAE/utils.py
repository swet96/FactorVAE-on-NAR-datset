import torch
import pickle
import torch.nn.functional as F


def save(path: str, obj) -> None:
    """
    Save model to path
    """
    outfile = open(path, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


def load(path: str):
    """
    Load model from path
    """
    infile = open(path, 'rb')
    obj = pickle.load(infile)
    infile.close()

    return obj


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
