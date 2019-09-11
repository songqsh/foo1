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

        # kernel
        self.fc1 = nn.Linear(2, 1)
        self.fc2 = lambda x: torch.tensor([x, x**2])
        
    def forward(self, x):        
        #layers
        x = self.fc2(x) #visible layer
        x = self.fc1(x) #hidden layer
 
        return x
    
    
#loss function 
criterion = nn.MSELoss()

#optimizer

net = Net()

# Train the model



# target function
a = 1.
b = 2.
c = -1.
f = lambda x: a*x**2+b*x+c


#training data

batch_size = 1000

x_train = torch.randn(batch_size,1)
y_train = f(x_train)


num_epochs = 10000

for epoch in range(num_epochs):

    # Forward pass
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    learning_rate = 0.001
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 
                                                    num_epochs, loss.item()))
        
#test
x_ = torch.randn(100,1)
y_ = f(x_)
plt.scatter(x_.detach().numpy(), y_.detach().numpy(), label='true')

y_pred = net(x_,2)
plt.scatter(x_.detach().numpy(), y_pred.detach().numpy(), label='pred')

plt.legend()
plt.show()