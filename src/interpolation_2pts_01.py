# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn

#import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input/output dim
        in_dim = 1
        out_dim = 1
        
        # kernel1
        self.fc1 = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        #layers
        x = self.fc1(x)
 
        return x
    
    
#loss function 
criterion = nn.MSELoss()

# target function
f = lambda x: np.abs(x)



#optimizer
net = Net()
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)



# Train the model
batch_size = 2

x_train = torch.randn(batch_size,1)
y_train = f(x_train)


num_epochs = 1000

for epoch in range(num_epochs):

    # Forward pass
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 
                                                    num_epochs, loss.item()))
        
#test

y_ = f(x_train)
plt.scatter(x_train.detach().numpy(), y_.detach().numpy(), label='true')

x_test=torch.linspace(-2, 2, 20).resize(20,1)
y_pred = net(x_test)
plt.plot(x_test.detach().numpy(), y_pred.detach().numpy(), label='pred')

plt.legend()
plt.show()