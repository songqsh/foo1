# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input/output dim
        in_dim = 1
        out_dim = 1
        
        # kernel1
        self.fc1 = nn.Linear(in_dim, 6)
        self.fc2 = nn.Linear(6, 6)
        
        #kernel2
        self.fc3 = nn.Linear(6, out_dim)
        
    def forward(self, x, m = 1):
        #layers
        if m==1:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.fc3(x)
        else:
            x = F.dropout(F.sigmoid(self.fc1(x)), p=0.5)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
 
        return x
    
    
#loss function 
criterion = nn.MSELoss()

#optimizer

net = Net()

# Train the model

num_epochs = 50000

# target function
a = 1.
b = 2.
f = lambda x: a*x**2+b


#training data

batch_size = 1000

x_train = torch.randn(batch_size,1)
y_train = f(x_train)



for epoch in range(num_epochs):

    # Forward pass
    outputs = net(x_train, 2)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    learning_rate = 0.001
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 500 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 
                                                    num_epochs, loss.item()))
        
#test
x_ = torch.randn(100,1)
y_ = f(x_)
plt.scatter(x_.detach().numpy(), y_.detach().numpy(), label='true')

y_pred = net(x_,2)
plt.scatter(x_.detach().numpy(), y_pred.detach().numpy(), label='pred')

plt.legend()