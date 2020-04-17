# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 00:37:24 2018

@author: Mariam
"""
# Imports here
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import json


def load_model(arch, checkpoint_path):
    #Load checkpoint
    cp = torch.load(checkpoint_path)

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

    # Load a checkpoint and rebuild the model
    model.load_state_dict(cp['state_dict'])
    model.class_to_idx = cp['class_to_idx']
    return model

def process_image(im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process image for use in a PyTorch model
    size = 256, 256
    im.thumbnail(size)

    width, height = im.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    cropped_im = im.crop((left, top, right, bottom))

    np_im = np.array(cropped_im)

    np_im = np_im / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_im = (np_im - mean)/std

    np_im = np_im.transpose((2,0,1))

    return np_im


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    plt.title(title)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, tk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Open and process image
    im = Image.open(image_path)
    im = process_image(im)

    #Convert numpy array to Tensor
    im = torch.from_numpy(im).type(torch.FloatTensor)
    #Turn to 4 dimensions for densenet to work properly (0,3,224,224)
    im.unsqueeze_(0)
    #Pass through model
    logps = model(im)
    #Convert to scores
    ps = torch.exp(logps)
    #5 top classes
    top_p, top_class = ps.topk(tk, dim=1)
    
    #Convert to a hashable object for use with dictionary
    top_p = top_p.detach().numpy()[0]
    top_class = top_class.detach().numpy()[0]

    top_p = top_p.tolist()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in top_class]

    return top_p, classes


#Open flower category names file
json_path = input('Please, Enter the JSON filename:  ')
with open(json_path, 'r') as f:
    cat_to_name = json.load(f)

print("JSON file included successfuly, Please choose a picture through the open browser.")
#Adjust directories
parent_dir = 'E:/Courses'+'/flower_data'
train_dir = parent_dir + '/train'
valid_dir = parent_dir + '/valid'
checkpoint_path = 'checkpoint.pth'


# Load trained model checkpoint
model = load_model('densenet', checkpoint_path)
model.eval()

# Open a dialogue for user to choose an image
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename()


# Display an image along with the top 5 classes
plt.figure(figsize = (4,9))
ax = plt.subplot(2,1,1)
im = Image.open(image_path)
img = process_image(im)
imshow(img, ax);

top_p, classes = predict(image_path, model)
top_class = [cat_to_name[i] for i in classes]

#Reverse so high probability comes on top
top_class.reverse()
top_p.reverse()

plt.subplot(2,1,2)
plt.barh(y = top_class, width = top_p, height = 0.5)
plt.show()
