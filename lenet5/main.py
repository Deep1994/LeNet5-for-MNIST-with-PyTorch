# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:56:46 2019

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from lenet5 import LeNet5


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
batch_size = 128
learning_rate= 0.001
num_epochs = 10


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/mnist', 
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),
                                                   transforms.ToTensor()]))

test_dataset = torchvision.datasets.MNIST(root='./data/mnist', 
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                                   transforms.Resize((32, 32)),
                                                   transforms.ToTensor()]))

# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False)

# take a look at the dataset
x, label = iter(train_loader).next()
print('x: ', x.shape) 
print('label: ', label.shape)

model = LeNet5().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(model)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += torch.eq(predicted, labels).float().sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')