#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import random
import torchvision
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def gen_data_with_labels(shape, add_signal = False, SNR = 1):

    #generate noise
    noise_layers = []
    df = 2
    for i in range(shape[0]):
        noise_layers.append(np.array(chi2.rvs(df, size = shape)))
    noise = np.stack(noise_layers)
    normalized_noise = noise/np.amax(noise)

    #generate signal
    signal = np.zeros(shape)
    row = np.random.randint(0, shape[-1])
    signal[row] = 1/SNR

    #combine
    if add_signal:
      noisy_signal = (signal+normalized_noise)/np.amax(signal+normalized_noise)
      label = 1
    else:
      noisy_signal = normalized_noise
      label = 0
    return noisy_signal, label

#show the noise signal and the signal
x,y = gen_data_with_labels((20,20), add_signal = add_signal, SNR=3)
fig, ax = plt.subplots(1,1, figsize = (10,5))
ax.imshow(x)
print(y)

# %%
#generate 1000 pieces of data
x_data,y_data = [],[]
shape= (20,20)
for i in range(1000):
    add_signal = random.choice([True, False])
    input_img, label = gen_data_with_labels(shape, add_signal = add_signal, SNR=4)
    x_data.append(input_img)
    y_data.append(label)

#SCALING THE DATA
x_data = np.array(x_data)
y_data = np.array(y_data)
print(f'shape of x: {x_data.shape}, \nshape of y: {y_data.shape}')
#scaling the images
# %%
#CREATING THE DATALOADER
from torch.utils.data import Dataset, DataLoader
class DatasetClass(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length

trainset = DatasetClass(x_data,y_data)
trainloader = DataLoader(trainset,batch_size= 10,shuffle = True)
# %%
#DEFINE THE NEURAL NETWORK
from torch import nn
from torch.nn import functional as F 

class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        print('input shape', input_shape)
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64,1)
        # self.fc3 = nn.Linear(64,1)

    def forward(self,x):
        # print('1',x.shape)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        # print('2',x.shape)
        x = F.sigmoid(self.fc2(x))
        # # print('3',x.shape)
        # x = torch.sigmoid(self.fc3(x))
        # print('4',x.shape)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_shape):
        super(ConvNet, self).__init__()
        print('input shape', input_shape)
        self.fc1 = nn.Conv2d(1, 6, 3,1)
        self.fc2 = nn.Conv2d(6,3,3,1)

        self.fc3 = nn.Linear(3*3*3, 16)
        self.fc4 = nn.Linear(16,1)
    def forward(self,x):
        # print('1',x.shape)
        x = F.relu(self.fc1(x))
        # print('2',x.shape)
        x = F.max_pool2d(x,2,2)
        # print('3',x.shape)
        x = F.relu(self.fc2(x))
        # print('4',x.shape)
        x = F.max_pool2d(x,2,2)
        # print('5',x.shape)
        x = x.view(-1,3*3*3)
        # print('6',x.shape)
        x = F.relu(self.fc3(x))
        # print('7', x.shape)
        x = F.sigmoid(self.fc4(x))
        # print('8', x.shape)
        return x
# %%
#DEFINE MODEL PARAMETERS
lr = 0.001
epochs = 1000

model = Net(input_shape = x_data.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss_fn = nn.BCELoss()

# %%
#TRAINING LOOP
accuracy, losses = [],[]

for i in range(epochs):
  for j, (x_train, y_train) in enumerate(trainloader):
    #calculate output
    x_train = x_train.view(10,1,20,20)
    # print(x_train.shape, 'X-train shape')
    output = model(x_train)
    #calculate loss
    loss = loss_fn(output, y_train.reshape(-1,1))
    # print('LOSS',type(loss), loss)
    #accuracy
    predicted = torch.max(output.data,1)[1]
    acc = (predicted==y_train).sum()

    #backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if i%10==0:
    losses.append(loss)
    accuracy.append(acc)
    print(f'epoch:{i}, loss:{loss}, accuracy: {acc}')
# %%
#LET'S TEST THE MODEL
test_x, test_y = [],[]
shape= (20,20)
for i in range(100):
    add_signal = random.choice([True, False])
    input_img, label = gen_data_with_labels(shape, add_signal = add_signal, SNR=3)
    test_x.append(input_img)
    test_y.append(label)

#SCALING THE DATA
test_x = np.array([np.ravel(sc.fit_transform(x)) for x in test_x])
test_y = np.array(test_y)
print(f'shape of x: {test_x.shape}, \nshape of y: {test_y.shape}')

testset = DatasetClass(test_x,test_y)
testloader = DataLoader(testset,batch_size= 1,shuffle = True)
# %%
with torch.no_grad():
  correct = 0

  for x_test, y_test in testloader:
    plt.imshow(x_test.reshape(20,20))
    y_val = model(x_test)
    # print(y_val, 'Y_VAL')
    # predicted = torch.max(y_val, 1)[1]
    predicted = model(torch.tensor(x_test, dtype=torch.float32))
    correct += (predicted==y_test)
    print(f'predicted:{predicted}, \ny_test:{y_test}, \ny_val: {y_val}')
    break

# %%
