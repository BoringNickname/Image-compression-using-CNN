#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
#DATA GENERATROR

def gen_data_with_varied_strengths_and_mixed_signals(shape=(20,20), SNR = 1, n_samples=1e3):
    x_data, y_data  = [], []
    for i in range(n_samples):
                
        #generate noise
        noise = np.array(chi2.rvs(df = 2, size = shape))
        normalized_noise = noise/np.amax(noise)
                    
        #generate signal
        add_signal = random.choice([True, False]) #choose whether an example will have signal in it
        signal_strength = np.random.rand()
        signal = np.zeros(shape)
        row = np.random.randint(0, shape[-1]) #pick a random row
        signal[row] = signal_strength
        
        #combine
        if add_signal:
            noisy_signal = (signal+normalized_noise)
            ref_img = signal
        else:
            noisy_signal = normalized_noise
            ref_img = np.zeros(shape)
        
        x_data.append(noisy_signal)
        y_data.append(ref_img)
    return x_data, y_data
#SHOW A NICE SUBPLOT GRID OF THE GENERATED DATA
if True:
  x_data, y_data = gen_data_with_varied_strengths_and_mixed_signals((20,20), n_samples=9, SNR=2)
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

# %%
#CREATING A CUSTOM TRAINLOADER
class DatasetClass(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
    # self.y.to(device)
    # self.x.to(device)
    
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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cl1 = nn.Conv2d(1,10,5,1, padding=2)
        self.fc1 = nn.Linear(10*int(shape[0]/2)**2, shape[0]**2)
        
    def forward(self,x):
        x = x.view(batch_size, 1, *shape)
        x = F.relu(self.cl1(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = x.view(batch_size, *shape)
        return x

class SemanticModel(nn.Module):
    def __init__(self):
        super(SemanticModel, self).__init__()
        self.cl1 = nn.Conv2d(1,5,5,1, padding = 2)
        self.cl2 = nn.Conv2d(5,10,5,1, padding = 2)

        self.fc1 = nn.Linear(10*7*7, shape[0]*shape[1])

    def forward(self,x):
        x = x.view(batch_size, 1, *shape)
        x = F.relu(self.cl1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.cl2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = x.view(batch_size,*shape)
        return x
# %%
#SOME GLOBAL VARIABLES
lr = 0.004
epochs = 50
batch_size = 100
shape = (30,30)

#CREATING THE DATA
data = DatasetClass(*gen_data_with_varied_strengths_and_mixed_signals(shape, SNR=2, n_samples = 10000)) #DATA IS CHANGED!!
trainloader, testloader = train_test_split_dataloaders(data, 0.8)


# %%
accuracy, losses = [],[]
model = ConvNet()
# model.to(device)
# model = SemanticModel()
print(model)

#OPTIMIZER AND LOSS FUNCTION
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.BCELoss()

#TRAIN THE MODEL
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
#TEST THE MODEL
with torch.no_grad():
    n = 0
    for x_test, y_test in testloader:
        predicted = model(x_test)
        fig, ax = plt.subplots(3,3, figsize = (11,10))
        for row in ax:
            row[0].imshow(x_test[n], vmin=0, vmax=1)
            row[1].imshow(predicted[n], vmin=0, vmax=1)
            row[2].imshow(y_test[n], vmin=0, vmax=1)
            n+=1
        break
# %%