##code from: https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
to_pil_image = transforms.ToPILImage()


def image_to_vid(images, path):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(f'{path}generated_images.gif', imgs)


def save_reconstructed_images(recon_images, epoch, path):
    save_image(recon_images.cpu(), f"{path}output{epoch}.jpg")


def save_true_images(true_images, epoch, path):
    save_image(true_images.cpu(), f"{path}output{epoch}_true.jpg")


def save_loss_plot(train_loss, valid_loss, path):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{path}loss.jpg')
    plt.show()

