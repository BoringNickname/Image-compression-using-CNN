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
from scipy.stats import chi2


# %%
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
    df = 1
    noise_layers = []
    for i in range(shape[0]):
        noise_layers.append(np.array(chi2.rvs(df, size = shape[0])))
    noise = np.stack(noise_layers)/np.amax(np.stack(noise_layers))
    noisy_signal = arr+noise
    return noisy_signal/np.amax(noisy_signal)

def gen_data(shape, n_datapoints):
    x_data, y_data = [], []
    for i in range(n_datapoints):
        signal = generate_signals(shape)
        y_data.append(signal.astype(np.float32))
        x_data.append(generate_noise_templates(signal, shape).astype(np.float32))
    return x_data, y_data

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
        
        self.enc1 = nn.Conv2d(1,5,5,1, padding = 2)
        self.enc2 = nn.Conv2d(5,15,5,1, padding = 2)
        
        self.fc1 = nn.Linear(5*5*15, 100)
        self.fc2 = nn.Linear(100,20)
        self.fc3 = nn.Linear(20,2)

        self.dc1 = nn.Linear(2,20)
        self.dc2 = nn.Linear(20,100)
        self.dc3 = nn.Linear(100, 20*20)

    def forward(self,x):
        x = F.relu(self.enc1(x))
        x = F.max_pool2d(x, 2,2)
        x = F.relu(self.enc2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 5*5*15)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.dc1(x))
        x = F.relu(self.dc2(x))
        x = F.relu(self.dc3(x))
        print(x.shape)
        return x.view(5,1,20,20)

def test_image_reconstruction(net, testloader):
    
    for image in testloader:
        img, _ = image
        img = img.to(device)
        img = torch.flatten(img)
        outputs = net(img)
        outputs = outputs.view(*shape)
        save_image(outputs, 'spectrogram.png')
        break
# %%


model = Autoencoder()

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
            y = self.transform(y)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

# Let's create 10 RGB images of size 128x128 and 10 labels {0, 1}
data, targets = gen_data((20,20), 100)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor()
    ])

dataset = MyDataset(data, targets, transform=transform)
trainloader = DataLoader(dataset, batch_size=5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
#%%

import time
start_time = time.time()

for i in range(5):
    
    trn_corr, tst_corr = 0,0
    
    for b, (x_train, y_train) in enumerate(trainloader):
        b+=1
        print('X_TRAIN',x_train.shape, 'Y_TRAIN',y_train.shape)
        y_pred = model(x_train) #no flattening
        
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted==y_train).sum()
        trn_corr += batch_corr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b%100==0:
            print(f'EPOCH: {i}, BATCH: {b}, LOSS: {loss.item()}')
            
    train_loss.append(loss)
    train_correct.append(trn_corr)
    
    #TEST
    with torch.no_grad():
        for b,(x_test, y_test) in enumerate(trainloader):
            y_val = model(x_test)
            
            predicted = torch.max(y_val.data,1)[1]
            tst_corr += (predicted==y_test).sum()
            
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)
        
current_time = time.time()
total - current_time/start_time
print(f'Time passed: {total/60}')
# %%

# %%
