# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input/output dim
        in_dim = 1
        out_dim = 1
        
        # kernel1
        H = 10
        self.fc1 = nn.Linear(in_dim, H)
        self.fc2 = nn.Linear(H, out_dim)
        
    def forward(self, x, m = 1):
        #layers
        if method == 1:
            x = self.fc1(x) 
            x = F.sigmoid(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            
        return x
    
    
#loss function 
criterion = nn.MSELoss()

# target function
f = lambda x: np.abs(x)



#optimizer
net = Net()
learning_rate = 0.1
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)




#train data
batch_size = 3
#x_train = 2*torch.rand(batch_size,1)-1
x_train = torch.tensor([-.5, 0., .5]).reshape(3,1)
y_train = f(x_train)

# Train the model
num_epochs = 20000
m = 1

for epoch in range(num_epochs):

    # Forward pass
    outputs = net(x_train, m)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2000 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 
                                                    num_epochs, loss.item()))
        
#print parameters
#print model parameters automatically
#for p in net.parameters():
#  print(p)        
        
#test
y_ = f(x_train)
plt.scatter(x_train.detach().numpy(), y_.detach().numpy(), label='true')

x_test=torch.linspace(-1, 1, 100).reshape(100,1)
y_pred = net(x_test, m)
plt.plot(x_test.detach().numpy(), y_pred.detach().numpy(), label='pred')

plt.legend()
plt.show()