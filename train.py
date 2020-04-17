# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:45:20 2018

@author: Mariam
"""
# Imports here
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

#Adjust directories
parent_dir = 'E:/Courses'+'/flower_data'
train_dir = parent_dir + '/train'
valid_dir = parent_dir + '/valid'
checkpoint_path = 'checkpoint_test.pth'

#Adjust architectures and parameters
arch = 'densenet'
ler = 0.001
epochs = 2

# Define transforms for the training and validation sets
train_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.RandomRotation(90),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomResizedCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Using the image datasets and the tranforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)

# Build and train your network

#Pretrained Model
if (arch == 'densenet'):
    model = models.densenet121(pretrained=True)
    feat_inp = 1024
elif (arch == 'alexnet'):
    model = models.alexnet(pretrained=True)
    feat_inp = 9216
else:
    print("Sorry, we don't have this model")
    exit(0)

#Freeze features
for par in model.parameters():
    par.requires_grad = False


#Modify fully connected part
classifier = nn.Sequential(OrderedDict([
                          ('h1', nn.Linear(feat_inp, 512)),
                          ('relu', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.5)),
                          ('h2', nn.Linear(512, 102)),
                          ('drop2', nn.Dropout(p=0.5)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

"""
#Load Checkpoint for further training
cp = torch.load('checkpoint.pth')
model.load_state_dict(cp['state_dict'])
"""

#Criterion & Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), ler)


"""
#Load optimizer state for further training
optimizer.load_state_dict(cp['optimizer'])
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
"""

#Some training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Loop
for e in range(epochs):
    train_loss = 0
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        #loss = 0
        #loss.requires_grad = True
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    #Validation Loop
    with torch.no_grad():
        accuracy = 0
        valid_loss = 0
        model.eval()
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            outputs = torch.exp(outputs)
            top_p, top_class = outputs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        print("Accuracy: {:.3f}\t\t".format(accuracy/len(validloader)), "Train Loss: {:.3f}\t".format(train_loss/len(trainloader)), "Validation Loss: {:.3f}".format(valid_loss/len(validloader)))
    
# Save the checkpoint
model.class_to_idx = train_datasets.class_to_idx
model.to("cpu")
torch.save({'arch': arch,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optimizer': optimizer.state_dict()},
            checkpoint_path)