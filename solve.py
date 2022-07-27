"""
Implementation of Genetic Algorithm to solve the NAR problem
"""
import os

import torch
import pygad

from dataloader import load_nar
from model import FactorVAEGenetic, Discriminator

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} as the accelerator:)")

ckpt_dir = "./checkpoints/"
ckptname="nar_fvae_copy.pth.tar"
logger_filename= f"./logger/nar_fvae_copy.log"

def load(filepath):
    with open(filepath, 'rb+') as f:
        states = torch.load(f)
    D = Discriminator(z_dim=10).to(device)
    VAE = FactorVAEGenetic().to(device)
    D.load_state_dict(states["model_states"]["D"])
    VAE.load_state_dict(states["model_states"]["VAE"])
    iter = states["iter"]

    return D, VAE, iter

train_dataloader, _ = load_nar(num_task=10, batch_size=1)

D, VAE, iter = load(os.path.join(ckpt_dir, str(ckptname)))
print("Found checkpoints :)")

for i, (x_true, options, solution_from_dataset) in enumerate(train_dataloader):
    x_true = x_true.to(device)
    options = options.to(device)
    solution_from_dataset = solution_from_dataset.to(device)

    squeezed_z = VAE(x_true, no_dec=True)
    print(squeezed_z.shape)

    break

def fitness(genome, genome_idx):
    """
    Method to find fitness of a genome/ chromosome
    """
    # perturbation size
    genome = torch.from_numpy(genome).float()
    recon = VAE(x=None, z=(squeezed_z + genome))

    diffs = []
    for i in range(4):
        diff = torch.cdist(recon, options[:, i])
        diffs.append(diff)
    
    fit = -1 * max(diffs)

    return fit


fitness_function = fitness

num_generations = 100
num_parents_mating = 4

sol_per_pop = 5
num_genes = squeezed_z.shape[1]

init_range_low = 0
init_range_high = 0.04

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

perturbation = torch.Tensor(solution).reshape(1, -1)
print(perturbation)
solution = VAE(x=None, z=(squeezed_z + perturbation))

diffs = []
for i in range(4):
    diff = torch.cdist(solution, options[:, i])
    diffs.append(diff)

solution_option = diffs.index(max(diffs))
print(solution_from_dataset, solution)
