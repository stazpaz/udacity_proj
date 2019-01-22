#imports
import argparse
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms,models

from PIL import Image
import glob
import os

import json

def args_parser():
    parser = argparse.ArgumentParser(description='trainer file')
    
    parser.add_argument('--data_dir', type=str, default='flowers', help='file directory')
    parser.add_argument('--gpu', type=bool, default=True, help='True: gpu, False: cpu')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=[1000, 500], help='hidden units for layer')
    parser.add_argument('--epochs', type=int, default=2, help='training epochs')
    parser.add_argument('--arch', type=str, default='vgg16', choices = {'vgg16','resnet18','densenet161'}, help='select architecture: vgg16,resnet18,densenet161')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
    args = parser.parse_args()
    return args


def setup_classifier(model,hidden_sizes,arch):
    
    input_size = 0
    
    if arch == 'vgg16':
        input_size = model.classifier[0].in_features
    elif arch == 'resnet18':
        input_size = model.fc.in_features
    elif arch == 'densenet161':
        input_size = model.classifier.in_features
    else:
        input_size = 0
        
    print("input size" , input_size)    
    print(hidden_sizes)
    output_size = 102
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([ 
        ('fc1', nn.Linear(input_size, hidden_sizes[0])), 
        ('relu1', nn.ReLU()), 
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])), 
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),
        ('output', nn.Linear(hidden_sizes[1], output_size)), 
        ('softmax', nn.LogSoftmax(dim=1)) 
    ]))
    model.classifier = classifier
    print("_______________________________")
    print(model.classifier)
    print("_______________________________")
    return model

# Implement a function for the validation pass
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        #images, labels = images.to('cuda'), labels.to('cuda')
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def train_model(epochs,trainloader, testloader, device, model, optimizer, criterion):
    print_every = 40
    steps = 0
    running_loss = 0
    print("training the model")
    # change to cuda
    print(device)
    model.to(device)
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion,device)
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                    running_loss = 0
    return model
    
def validate_test_set(model, testloader, device,criterion):                    
    # TODO: Do validation on the test set
    correct = 0
    total = 0
    
    "______________MODEL_________________"
    print(model)
    "______________MODEL_________________"
    #model.to('cuda')
    model.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            #inputs, labels = inputs.to('cuda'), labels.to('cuda')
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def save_checkpoint(image_trainset,model,arch):
    '''
    checkpoint = {'input_size': (3, 224, 224),
              'output_size': 102,
              'hidden_layers': hidden_layers,
              'batch_size': batch_size,
              'learning_rate': lr,
              'model_name': model_name,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'epoch': epochs,
              'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
    
    '''
    checkpoint = {'input_size': 25088,
              #'class_to_idx': model.class_to_idx
              'class_to_idx' : image_trainset.class_to_idx,
              'classifier': model.classifier,
              'output_size': 102,
              'hidden_sizes': [1000,500],
              'epocs':5,
              'arch': arch,
              'state_dict': model.classifier.state_dict()
    }
    torch.save(checkpoint, 'checkpoint.pth')
    
    print("checkpoint saved!")

def main():
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([ 
        transforms.RandomRotation(30), 
        transforms.Resize(256),
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    test_transforms = transforms.Compose([ 
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # TODO: Load the datasets with ImageFolder
    image_trainset = datasets.ImageFolder(train_dir,transform = training_transforms)
    image_testset = datasets.ImageFolder(test_dir,transform = test_transforms)
    image_testset_notransform = datasets.ImageFolder(test_dir)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_trainset, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(image_testset, batch_size = 32)
    
    args = args_parser()
    print (args)
    #is_gpu=args.gpu

     
    # TODO: Build and train your network
    #model = models.vgg16(pretrained=True)
    #model = models(args.arch)
    #model_arch = args.arch
    #print (model_arch)
    
    #model = getattr(models,model_arch)
    #import torchvision
    model = getattr(torchvision.models, args.arch)(pretrained=True)
    #print(model)
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args.gpu)
    if(args.gpu == True):
        device = "cuda"
    else:
        device = "cpu"
    
    print(device)
    for param in model.parameters():
        param.requires_grad = False
        
    model = setup_classifier(model, args.hidden_units,args.arch)
    print(model)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    trmodel = train_model(args.epochs,trainloader, testloader, device, model, optimizer, criterion)
    validate_test_set(trmodel, testloader, device,criterion)
    #save_checkpoint(image_trainset,model.classifier,args.arch)
    save_checkpoint(image_trainset,model,args.arch) #image_trainset needed?????
    
if __name__ == '__main__': main()
