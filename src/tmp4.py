import torch
import torch.nn as nn



n_mesh = 4

row0 = [1.]+[0.]*4
row4 = row0[::-1]
row1 = [.5, -1., .5]+[0]*2
row2 = [0.]+[.5, -1, .5]+[0.]
row3 = row1[::-1]
A = torch.tensor([row0, row1, row2,row3,row4], dtype= torch.float).reshape(n_mesh+1, n_mesh+1)
b = torch.tensor([.25, 0.0625, 0.0625, 0.0625, 0.25], dtype=torch.float).reshape(n_mesh+1, 1)


x = torch.solve(b,A)
print(f'target is {x}')


#nn solver
net = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10,1),
)

def loss_list():  
  ix = torch.tensor(range(n_mesh+1), dtype=torch.float).reshape(n_mesh+1,1)
  return A@net(ix)-b


print_n = 100
n_epoch=500000; epoch_per_print= int(n_epoch/print_n)

for epoch in range(n_epoch):
    #ipdb.set_trace()
    loss = sum([a**2. for a in loss_list()]) #forward pass
    #backward propogation
    lr = max(1./(n_epoch+10), 0.001)
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum = .8) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % epoch_per_print == 0:
        x_cod = torch.tensor(range(n_mesh+1), dtype=torch.float).reshape(n_mesh+1,1)
        x_pre = net(x_cod).reshape(n_mesh+1).tolist()  
        x_pre = ["%2f"%c for c in x_pre]        
        print(f'Epoch: {epoch+1}/{n_epoch}, Loss: {loss.item()}, \n soln: {x_pre}')
        
        
        
