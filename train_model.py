# @Time : 2022/4/6 14:03
# @Author : Zg
import torch
from utils.dataloaders import get_mnist_dataloaders
from jointvae.models import VAE
from torch import optim
from jointvae.training import Trainer
from viz.visualize import Visualizer
import matplotlib.pyplot as plt

train_loader, test_loader = get_mnist_dataloaders(batch_size=256)

# Latent distribution will be joint distribution of 10 gaussian normal distributions
# and one 10 dimensional Gumbel Softmax distribution
latent_spec = {'cont': 10,
               'disc': [10]}
model = VAE(latent_spec=latent_spec, img_size=(1, 32, 32), use_cuda=True).cuda()

# Build optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Define the capacities
# Continuous channels
cont_capacity = [0.0, 5.0, 25000, 30.0]  # Starting at a capacity of 0.0, increase this to 5.0
                                         # over 25000 iterations with a gamma of 30.0
# Discrete channels
disc_capacity = [0.0, 5.0, 25000, 30.0]  # Starting at a capacity of 0.0, increase this to 5.0
                                         # over 25000 iterations with a gamma of 30.0

# Build a trainer
trainer = Trainer(model, optimizer,
                  cont_capacity=cont_capacity,
                  disc_capacity=disc_capacity)

# Build a visualizer which will be passed to trainer to visualize progress during training
viz = Visualizer(model)

# Train model for 10 epochs
# Note this should really be a 100 epochs and trained on a GPU, but this is just to demo

trainer.train(train_loader, epochs=5, save_training_gif=('./training.gif', viz))
torch.save(model.state_dict(), "trained_models/test/gmm_model_2.pt")

