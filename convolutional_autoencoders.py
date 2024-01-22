#%%
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

num_epochs = 50
learning_rate = 1e-3
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

trainset = datasets.CIFAR10(
    root='./data',
    train=True, 
    download=True,
    transform=transform
)
testset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
trainloader = DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True
)
testloader = DataLoader(
    testset, 
    batch_size=batch_size, 
    shuffle=True
)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
def make_dir():
    image_dir = 'Conv_CIFAR10_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
def save_decoded_image(img, name):

    img = img.view(img.size(0), 3, 32, 32)
    save_image(img, name)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        #encoder
        self.enc1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3)
        
        self.enc2 = nn.Conv2d(in_channels = 8, out_channels = 4, kernel_size = 3)

        #decoder
        self.dec1 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size = 3)

        self.dec2 = nn.ConvTranspose2d(in_channels=8, out_channels = 3, kernel_size = 3)

    def forward(self,x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x

def train(net, trainloader, num_epochs):
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss/len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, num_epochs, loss))

        if epoch % 10 == 0:
            save_decoded_image(img.cpu().data, name = './Conv_CIFAR10_Images_original{}.png'.format(epoch))
            save_decoded_image(outputs.cpu().data, name='./Conv_CIFAR10_Images_decoded{}.png'.format(epoch))

    return train_loss

def test_image_reconstruction(net,testloader):
    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        outputs = net(img)
        outputs = outputs.view(outputs.size(0), 3, 32).cpu().data
        save_image(outputs, 'conv_cifar10_reconstructed_image.png')
        break

net = Autoencoder()
print(net)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)

device = get_device()
print(device)
net.to(device)
train_loss = train(net, trainloader, num_epochs)

plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
test_image_reconstruction(net, testloader)

# %%
