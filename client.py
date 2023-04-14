from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from net import Net

class Client:

    def __init__(self, recievedModel, seed=1):
        torch.manual_seed(seed)
        self.model = Net()
        if recievedModel:
            self.model.load_state_dict(recievedModel)

    def updateModel(self, recievedModel):
        self.model.load_state_dict(recievedModel)

    def getModel(self):
        return self.model.state_dict()
        
    def train(self, device, train_loader, optimizer, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            log_interval = 100000
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                return


    def test(self, device, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    def run(self, epochs, batch):

        use_cuda = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available()

        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
            train_kwargs.update(cuda_kwargs)

        train_kwargs = {'batch_size': batch}
        

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

        lr = 1.0

        self.model = self.model.to(device)
        optimizer = optim.Adadelta(self.model.parameters(), lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        for epoch in range(1, epochs + 1):
            self.train(device, train_loader, optimizer, epoch)
            scheduler.step()