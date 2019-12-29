import torch
import torch.nn as nn


net = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10,1),
)

A = torch.tensor([3, 5, -2, 10], dtype=torch.float).reshape(2,2)
b = torch.ones(2,1)

x = torch.solve(b,A)
print(f'target is {x}')

def loss_list():  
  ix = torch.tensor([0,1], dtype=torch.float).reshape(2,1)
  return A@net(ix)-b


print_n = 10
n_epoch=50000; epoch_per_print= int(n_epoch/print_n)

for epoch in range(n_epoch):
    #ipdb.set_trace()
    loss = sum([a**2. for a in loss_list()]) #forward pass
    #backward propogation
    lr = max(1./(n_epoch+100), 0.001)
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum = .8) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % epoch_per_print == 0:
        x_pre = (net(torch.tensor([0,1], dtype=torch.float).reshape(2,1)))          
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, \n soln: [{x_pre[0].item()}, {x_pre[1].item()}]')