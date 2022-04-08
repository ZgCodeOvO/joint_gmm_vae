# @Time : 2022/4/6 14:03
# @Author : Zg

from utils.load_model import load
from viz.visualize import Visualizer as Viz
import matplotlib.pyplot as plt

path_to_model_folder = './trained_models/test/'
model = load(path_to_model_folder)

print(model.latent_spec)
print(model)

# Create a Visualizer for the model
viz = Viz(model)
viz.save_images = False  # Return tensors instead of saving images

# traversals = viz.all_latent_traversals()
# plt.imshow(traversals.numpy()[0, :, :], cmap='gray')
# plt.show()

# Traverse 3rd continuous latent dimension across columns and first
# discrete latent dimension across rows
traversals = viz.latent_traversal_grid(cont_idx=5, cont_axis=1, disc_idx=0, disc_axis=0, size=(10, 10))
plt.imshow(traversals.numpy()[0, :, :], cmap='gray')
plt.show()














