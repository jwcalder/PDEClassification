import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pde import heat_eq_solver
from pde import wave_eq_solver
from pde import random_source
from scipy import ndimage
import sys


class Net(nn.Module):
    def __init__(self,n,w=(32,64)):
        super(Net, self).__init__()
        m = n>>5
        self.conv1 = nn.Conv2d(2, w[0], 3, 1,padding=1)
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1,padding=1)
        self.fc1 = nn.Linear(w[1]*m*m, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # images start out nxn
        x = self.conv1(x)  #nxn images
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #n/4 x n/4 images
        x = self.conv2(x)  
        x = F.relu(x)
        x = F.max_pool2d(x, 8)  #n/16 x n/16 images
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#Training settings
n = 32
num_train = 64
sigma = 3 #Gaussian smoothing of source terms
cuda = True   #Use GPU acceleration 
batch_size = min(32,num_train)
learning_rate = 1    #Learning rate
epochs = 10000

#Solvers
wave = wave_eq_solver()
heat = heat_eq_solver(n)

#Training data
f = random_source(num_train>>1,n,sigma)
u_wave = wave.solve(f)
wave_data = np.stack((u_wave,f),axis=1)
wave_labels = np.zeros(num_train>>1)

u_heat = heat.solve(f)
heat_data = np.stack((u_heat,f),axis=1)
heat_labels = np.ones(num_train>>1)

train_data = torch.from_numpy(np.vstack((wave_data,heat_data))).float()
train_target = torch.from_numpy(np.hstack((wave_labels,heat_labels))).long()

#Shuffle
perm = torch.randperm(num_train)
train_data = train_data[perm]
train_target = train_target[perm]

#Testing data
num_test = 1000
f = random_source(num_test>>1,n,sigma)
u_wave = wave.solve(f)
wave_data = np.stack((u_wave,f),axis=1)
wave_labels = np.zeros(num_test>>1)

u_heat = heat.solve(f)
heat_data = np.stack((u_heat,f),axis=1)
heat_labels = np.ones(num_test>>1)

test_data = torch.from_numpy(np.vstack((wave_data,heat_data))).float()
test_target = torch.from_numpy(np.hstack((wave_labels,heat_labels))).long()

use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net(n).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

#Training epochs
wave = wave_eq_solver()
heat = heat_eq_solver(n)
for epoch in range(epochs):

    #Loop over minibatches
    model.train()
    for i in range(0,num_train,batch_size):

        #Convert to torch and put on device
        data = train_data[i:i+batch_size].to(device)
        target = train_target[i:i+batch_size].to(device)

        #Optimization step
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    #Check accuracy
    model.eval()
    with torch.no_grad():
        output = model(train_data.to(device)).cpu()
        train_loss = F.nll_loss(output, train_target)
        pred = torch.argmax(output,axis=1)
        train_acc = torch.mean((pred == train_target).float())
        output = model(test_data.to(device)).cpu()
        test_loss = F.nll_loss(output, test_target)
        pred = torch.argmax(output,axis=1)
        test_acc = torch.mean((pred == test_target).float())
        print('Epoch %d, Train Loss = %f, Train Accuracy = %f, Test Accuracy = %f'%(epoch,loss,100*train_acc,100*test_acc))














