import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib
# import matplotlib.pyplot as plt
import pickle as pk
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, save_true_images, save_input_images, save_cat_images, image_to_vid, save_loss_plot


matplotlib.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = model.ConvVAE([64, 60], [32, 15]).to(device)
# set the learning parameters
lr = 0.001
epochs = 100
batch_size = 16
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
# ])
# training set and train data loader
# trainset = torchvision.datasets.MNIST(
#     root='../data', train=True, download=True, transform=transform
# )
symbols = ["601857", "600028"]

years = range(2008, 2022)
imgs = []
labels = []
for s in symbols:
    for y in years:
        f = open(f"./kchart/{s}_{y}_imgs.pk", "rb")
        imgs.extend((pk.load(f)))
        f = open(f"./kchart/{s}_{y}_labels.pk", "rb")
        labels.extend(pk.load(f))
        
# imgs = [img/255.0 for img in imgs]
# f = open("../data/labels.pk", "rb")
# labels = pk.load(f)
# plt.figure()
# plt.imshow(imgs[0])

convert_tensor = transforms.ToTensor()
# print(convert_tensor(imgs[0]).size())
imgs = [convert_tensor(img) for img in imgs]
labels = [convert_tensor(img) for img in labels]
data = list(zip(imgs, labels))

idx_train = round(0.8*(len(data)))
trainset = data[:idx_train]
# trainset = imgs[:40]
# trainset = [convert_tensor(img) for img in trainset]
# trainset = [torch.unsqueeze(img, 0) for img in trainset]
testset = data[idx_train:]
# testset = imgs[:-1]
# testset = [convert_tensor(img) for img in testset]
# testset = [torch.unsqueeze(img, 0) for img in testset]

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
# validation set and validation data loader
# testset = torchvision.datasets.MNIST(
#     root='../data', train=False, download=True, transform=transform
# )
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

train_loss = []
valid_loss = []
result_path = "./outputs/"
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images, labels, input_images = validate(
        model, testloader, testset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    # print("images for cat:", recon_images.shape, input_images.shape)
    if (epoch+1)%100 == 0:
        # save_reconstructed_images(recon_images, epoch+1, result_path)
        # save_true_images(labels, epoch+1, result_path)
        # save_input_images(input_images, epoch+1, result_path)
        save_cat_images(recon_images, labels, epoch+1, result_path)
    
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

# save the reconstructions as a .gif file
# result_path = "./outputs/"
image_to_vid(grid_images, result_path)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss, result_path)
# save the true label to disk
# save_true_images(labels, epoch+1, result_path)

print('TRAINING COMPLETE')