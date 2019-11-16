"""
Code is based on the original provided at https://github.com/pytorch/examples/tree/master/mnist
Customized paramters and network to check behaviour
Checkpoint added to resume training as necessary
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import os

LR = 0.001
MOMENTUM = 0.6
BATCH_SIZE = 64
TEST_BATCH_SIZE=1000
EPOCHS = 10
RESUME = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10, kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.droput = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.droput(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1,320)
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, epoch, device, optimizer):
    model.train()    
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, idx * len(data), len(train_loader.dataset),
            100. * idx / len(train_loader), loss.item()))
    return loss.item()

def test(model, test_loader, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
        
    loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
    .format(loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    return loss

def main():
    torch.manual_seed(0)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])),batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=False, transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])),batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    device = torch.device("cuda")

    model = Net().to(device)

    if RESUME and os.path.isfile('checkpoint.pth'):
        print("Model loaded from checkpoint\n")
        model.load_state_dict(torch.load('checkpoint.pth'))
        # model.eval()

    # optimizer = optim.SGD(model.parameters(), LR, momentum = MOMENTUM)
    # Adam works better here. Why?
    optimizer = optim.Adam(model.parameters(), LR, betas=(0.9,0.999))
    # train_loss = []
    # test_loss = []
    for e in range(EPOCHS):
        train(model, train_loader, e, device, optimizer)
        test(model, test_loader, device)
        # train_loss.append(train(model, train_loader, e, device, optimizer))
        # test_loss.append(test(model, test_loader, device))
        if e+1 % 10 == 0:
            torch.save(model.state_dict(),'checkpoint.pth')

    # plt.plot(train_loss,'r', label = 'Traininig Loss')
    # plt.plot(test_loss,'b', label = 'Test Loss')
    # plt.show()

if __name__ == '__main__':
    main()