# FactorVAE-on-NAR-datset

Implementation of Factor-VAE using Neural Analogical Reasoning Dataset. Available at: https://www.kaggle.com/datasets/gmshroff/few-shot-nar


python train.py - the program will look for saved checkpoints, if the checkpoint is found and the number of epochs is less than max_epochs(5000), it will run.

python infer.py - Running this file will randomly pick one image from test set; it will save the reconstructed image of the input image when latent dimension traversal is done. It will be saved in the ‘plot’ folder.  

python plot.py - it will plot the training, KL Divergence and reconstructed loss over the training epochs. 
