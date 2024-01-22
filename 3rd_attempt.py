#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import random
import torchvision
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F 
#DATA GENERATROR
def gen_data_with_labels(shape=(20,20), SNR = 1, n_samples=1e3):
    x_data, y_data  = [], []
    for i in range(n_samples):
        #generate noise
        add_signal = random.choice([True, False])
        noise = np.array(chi2.rvs(df = 2, size = shape))
        normalized_noise = noise/np.amax(noise)

        #generate signal
        signal = np.zeros(shape)
        row = np.random.randint(0, shape[-1])
        signal[row] = 1

        #combine
        if add_signal:
            signal_plus_noise= (signal+SNR*normalized_noise)
            noisy_signal = signal_plus_noise/np.amax(signal_plus_noise)
            label = 1
        else:
            noisy_signal = normalized_noise
            label = 0
        
        x_data.append(noisy_signal)
        y_data.append(label)
    return x_data, y_data

def gen_data_with_signals(shape=(20,20), SNR = 1, n_samples=1e3):
    x_data, y_data  = [], []
    for i in range(n_samples):
        #generate noise
        noise = np.array(chi2.rvs(df = 2, size = shape))
        normalized_noise = noise/np.amax(noise)

        #generate signal
        signal = np.zeros(shape)
        row = np.random.randint(0, shape[-1])
        signal[row] = 1

        #combine
        signal_plus_noise= (signal+SNR*normalized_noise)
        noisy_signal = signal_plus_noise/np.amax(signal_plus_noise)

        x_data.append(noisy_signal)
        y_data.append(signal)
    return x_data, y_data

def gen_data_with_mixed_signals(shape=(20,20), SNR = 1, n_samples=1e3):
    x_data, y_data  = [], []
    for i in range(n_samples):
        #generate noise
        add_signal = random.choice([True, False])
        noise = np.array(chi2.rvs(df = 2, size = shape))
        normalized_noise = noise/np.amax(noise)

        #generate signal
        signal = np.zeros(shape)
        row = np.random.randint(0, shape[-1])
        signal[row] = 1

        #combine
        if add_signal:
            signal_plus_noise= (signal+SNR*normalized_noise)
            noisy_signal = signal_plus_noise/np.amax(signal_plus_noise)
            ref_img = signal/np.amax(signal)
        else:
            noisy_signal = normalized_noise
            ref_img = np.zeros(shape)
        
        x_data.append(noisy_signal)
        y_data.append(ref_img)
    return x_data, y_data

def gen_data_with_varied_strengths(shape=(20,20), SNR = 1, n_samples=1e3):
    x_data, y_data  = [], []
    for i in range(n_samples):
        #generate noise
        signal_strength = np.random.rand()
        noise = np.array(chi2.rvs(df = 2, size = shape))
        normalized_noise = noise/np.amax(noise)

        #generate signal
        signal = np.zeros(shape)
        row = np.random.randint(0, shape[-1])
        signal[row] = signal_strength

        signal_plus_noise= (signal+normalized_noise)
        noisy_signal = signal_plus_noise/np.amax(signal_plus_noise)
        ref_img = signal/np.amax(signal)
 
        x_data.append(noisy_signal)
        y_data.append(ref_img)
    return x_data, y_data

def gen_data_with_varied_strengths_and_mixed_signals(shape=(20,20), SNR = 1, n_samples=1e3):
    x_data, y_data  = [], []
    for i in range(n_samples):
        #generate noise
        add_signal = random.choice([True, False])
        singal_strength = np.random.rand()

        noise = np.array(chi2.rvs(df = 2, size = shape))
        normalized_noise = noise/np.amax(noise)

        #generate signal
        signal = np.zeros(shape)
        row = np.random.randint(0, shape[-1])
        signal[row] = singal_strength

        #combine
        if add_signal:
            signal_plus_noise= (signal+normalized_noise)
            noisy_signal = signal_plus_noise
            ref_img = signal
        else:
            noisy_signal = normalized_noise
            ref_img = np.zeros(shape)
        
        x_data.append(noisy_signal)
        y_data.append(ref_img)
    return x_data, y_data
#SHOW A NICE SUBPLOT GRID OF THE GENERATED DATA
if True:
  x_data, y_data = gen_data_with_varied_strengths_and_mixed_signals((20,20), n_samples=10, SNR=2)
  a = 0
  fig, ax = plt.subplots(3,3, figsize=(9,9), sharex=True, sharey=True)
  for i in range(3):
      for j in  range(3):
          ax[i,j].imshow(x_data[a])
          a+=1

  fig, ax = plt.subplots(3,3, figsize=(9,9), sharex=True, sharey=True)
  a=0
  for i in range(3):
      for j in  range(3):
          ax[i,j].imshow(y_data[a])
          a+=1

print([np.sum(y) for y in y_data])
# %%
#CREATING A CUSTOM TRAINLOADER
class DatasetClass(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length

def train_test_split_dataloaders(dataset, frac):
    train = torch.utils.data.Subset(dataset, range(0, int(frac*len(dataset))))
    test = torch.utils.data.Subset(dataset, range(int(frac*len(dataset)), len(dataset)))

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)

    return trainloader, testloader

# %%
class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape**2, 100)
        self.fc2 = nn.Linear(100,10)
        self.fc3 = nn.Linear(10,1)

    def forward(self,x):
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # print('input shape', input_shape)
        self.cl1 = nn.Conv2d(1,10,5,1)

        self.fc1 = nn.Linear(10*8*8, 150)
        self.fc2 = nn.Linear(150,1)
    def forward(self,x):
        x = x.view(batch_size, 1, *shape)
        x = F.relu(self.cl1(x))
        x = F.max_pool2d(x,3,3)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class ComplexConvNet(nn.Module):
    def __init__(self):
        super(ComplexConvNet, self).__init__()
        # print('input shape', input_shape)
        self.cl1 = nn.Conv2d(1,10,5,1, padding = 2)
        self.cl2 = nn.Conv2d(10,5,3,1, padding = 1)

        self.fc1 = nn.Linear(5*7*7, 25)
        self.fc2 = nn.Linear(25,1)

    def forward(self,x):
        x = x.view(batch_size, 1, *shape)
        
        x = F.relu(self.cl1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.cl2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class SemanticModel(nn.Module):
    def __init__(self):
        super(SemanticModel, self).__init__()
        self.cl1 = nn.Conv2d(1,5,5,1, padding = 2)
        self.cl2 = nn.Conv2d(5,10,5,1, padding = 2)

        self.fc1 = nn.Linear(10*7*7, shape[0]*shape[1])

    def forward(self,x):
        # print('0', x.shape)
        x = x.view(batch_size, 1, *shape)
        # print('0.5', x.shape)
        x = F.relu(self.cl1(x))
        # print('1',x.shape)
        x = F.max_pool2d(x,2,2)
        # print('2',x.shape)
        x = F.relu(self.cl2(x))
        # print('3',x.shape)
        x = F.max_pool2d(x,2,2)
        # print('4', x.shape)
        x = x.view(batch_size, -1)
        # print('5', x.shape)
        x = F.sigmoid(self.fc1(x))
        # print('6', x.shape)
        x = x.view(batch_size,*shape)
        # print('7', x.shape)
        return x
# %%
#SOME GLOBAL VARIABLES
lr = 0.004
epochs = 30
batch_size = 10
shape = (30,30)

#CREATING THE DATA
# data = DatasetClass(*gen_data_with_labels(shape, SNR=2, n_samples = 1000))
data = DatasetClass(*gen_data_with_varied_strengths_and_mixed_signals(shape, SNR=2, n_samples = 1000)) #DATA IS CHANGED!!
trainloader, testloader = train_test_split_dataloaders(data, 0.8)

# %%
accuracy, losses = [],[]
# model = ComplexConvNet()
model = SemanticModel()
print(model)
#OPTIMIZER AND LOSS FUNCTION
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.BCELoss()

for i in range(epochs):
  for j, (x_train, y_train) in enumerate(trainloader):
    #calculate output
    optimizer.zero_grad()
    output = model(x_train)
    #calculate loss
    loss = loss_fn(output, y_train) #DROPPED Y_TRAIN RESHAPE(1,-1)!!
    #accuracy
    predicted = torch.max(output.data,1)[1]
    #backpropagation
    loss.backward()
    optimizer.step()
    
  if i%10==0:
    losses.append(loss)
    # accuracy.append(acc)
    print(f'epoch:{i}, loss:{loss}')
# %%
with torch.no_grad():
    n = 1
    for x_test, y_test in testloader:
        predicted = model(x_test)
        fig, ax = plt.subplots(1,3, figsize = (3,10), sharex=True, sharey=True)
        ax[0].imshow(x_test[n])
        ax[1].imshow(predicted[n])
        ax[2].imshow(y_test[n])
        break
# %%
plt.imshow(x_test[n])
# %%
plt.imshow()