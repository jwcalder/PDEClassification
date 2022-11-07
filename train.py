import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pde import heat_eq_solver
from pde import wave_eq_solver


class Net(nn.Module):
    def __init__(self,n,w=(32,64)):
        super(Net, self).__init__()
        m = n>>4
        self.conv1 = nn.Conv2d(1, w[0], 3, 1,padding=1)
        self.conv2 = nn.Conv2d(w[0], w[1], 3, 1,padding=1)
        self.fc1 = nn.Linear(w[1]*m*m, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # images start out nxn
        x = self.conv1(x)  #nxn images
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #n/4 x n/4 images
        x = self.conv2(x)  
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  #n/16 x n/16 images
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#Training settings
n = 32
cuda = True   #Use GPU acceleration 
batch_size = 32
learning_rate = 1    #Learning rate
epochs = 2000

use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net(n).to(device)
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

#Training epochs
wave = wave_eq_solver()
heat = heat_eq_solver(n)
for epoch in range(epochs):

    #Generate random training data batch
    f = 2*np.random.rand(batch_size,n,n)-1
    u_wave = wave.solve(f)
    #wave_data = np.stack((u_wave,f),axis=1)
    wave_data = u_wave[:,None,:,:]
    wave_labels = np.zeros(batch_size)

    u_heat = heat.solve(f)
    #heat_data = np.stack((u_heat,f),axis=1)
    heat_data = u_heat[:,None,:,:]
    heat_labels = np.ones(batch_size)

    data = np.vstack((wave_data,heat_data))
    target = np.hstack((wave_labels,heat_labels))

    #Convert to torch and put on device
    data_torch = torch.from_numpy(data).float().to(device)
    target_torch = torch.from_numpy(target).long().to(device)

    #Optimization setp
    optimizer.zero_grad()
    output = model(data_torch)
    loss = F.nll_loss(output, target_torch)
    loss.backward()
    optimizer.step()

    #Check accuracy
    with torch.no_grad():
        output = output.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        pred = np.argmax(output,axis=1)
        acc = np.mean(pred == target)
        print('Epoch %d, Batch Loss = %f, Batch Accuracy = %.2f'%(epoch,loss,100*acc))














