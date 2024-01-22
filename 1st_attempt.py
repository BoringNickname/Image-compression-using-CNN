#%%
"""Adapted from https://debuggercafe.com/implementing-deep-autoencoder-in-pytorch/"""
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from torchvision.utils import save_image
import os
#%%
def generate_signals(shape):
    """generates a noisy signal of a random freuency
    """
    signal = np.zeros(shape)
    row = np.random.randint(0, shape[-1])
    signal[row] = 1
    return signal

def generate_noise_templates(arr, shape):
    """
    returns a normalized noisy version of a given signal
    """
    noisy_signal = arr+np.random.rand(shape[0], shape[1])
    return noisy_signal/np.amax(noisy_signal)

def gen_data(shape):
    x_data, y_data = [], []
    for i in range(100):
        signal = generate_signals(shape)
        y_data.append(signal)
        x_data.append(generate_noise_templates(signal, shape))
    #generates a dataset of len=1000
    print(len(x_data))
    return x_data, y_data

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
def make_dir():
    image_dir = 'Spectograms'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
def save_decoded_image(img, epoch):
    img = img.view(*shape)
    save_image(img, './Sceptrogram{}.png'.format(epoch))

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        feats = len(np.ravel(x_data[0]))
        print(feats, 'FEATS!!!!')
        self.enc1 = nn.Linear(in_features = feats, out_features= int(feats/4))
        self.enc2 = nn.Linear(in_features = int(feats/4), out_features=shape[0])

    def forward(self,x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))

        return x

def train(net, trainloader, num_epochs):
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0
        for image in trainloader:
            img, _ = image
            img = img.to(device)
            img = torch.flatten(img)
            # print(len(img))
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss/len(trainloader)
        train_loss.append(loss)

        if epoch % 5 == 0:
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch, num_epochs, loss))
            save_decoded_image(outputs.cpu().data, epoch)

    return train_loss

def test_image_reconstruction(net, testloader):
    for image in testloader:
        img, _ = image
        img = img.to(device)
        img = torch.flatten(img)
        outputs = net(img)
        outputs = outputs.view(*shape)
        save_image(outputs, 'spectrogram.png')
#         plt.imshow(outputs.cpu())
        break
"""
This bit I am still working on. I want to be able to use DataLoader and Dataset functions to get everything working properly with pytorch"""

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            # x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
# data = MyDataset(x_data, y_data, transform = transform)
# trainloader = DataLoader(data, batch_size)

"""A seemingly easier way to create my own dataset
"""
#%%

num_epochs = 50
learning_rate = 6e-3
batch_size = 100

shape = (12,12)
x_data, y_data = gen_data(shape)

# print(x_data[0])
tensor_x = torch.Tensor(x_data) 
tensor_y = torch.Tensor(y_data)

my_dataset = TensorDataset(tensor_x,tensor_y) 
trainloader = DataLoader(my_dataset)
net = Autoencoder()
device = get_device()
# device = 'cpu'
print('DEVICE', device)
print(torch.cuda.is_available())
net.to(device)

make_dir()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

train_loss = train(net, trainloader, num_epochs)

plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

test_image_reconstruction(net, trainloader)
# %%
