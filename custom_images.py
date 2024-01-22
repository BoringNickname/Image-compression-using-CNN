#%%
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms, models
from PIL import Image
from torchvision.utils import save_image
import os
from scipy.stats import chi2

import warnings
warnings.filterwarnings('ignore')

# %%
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
     [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
     [0.229, 0.224, 0.225])
])

# %%
def gen_data_with_labels(shape, add_signal = False, SNR = 1):

    #generate noise
    noise_layers = []
    df = 1
    for i in range(shape[0]):
        noise_layers.append(np.array(chi2.rvs(df, size = shape[0])))
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

