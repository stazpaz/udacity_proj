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

#print("hello")

def args_parser():
    
    pa = argparse.ArgumentParser(description='predictor')
    pa.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint location')
    pa.add_argument('--gpu', type=bool, default=True, help='True: gpu, False: cpu')
    pa.add_argument('--top_k', type=int, default=3, help='top k classes')
    pa.add_argument('--img', type=str, required='True',help='location of image')
    pa.add_argument('--names_file',type=str,default='cat_to_name.json', help='names of plants')
    
    args = pa.parse_args()
    return args

def load_checkpoint(filepath):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    checkpoint = torch.load(filepath, map_location={'cuda:0': 'cpu'})
    print("------------")
    print(checkpoint)
    print("------------")   
    #print(model)
    
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    print("model", model)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier.load_state_dict = checkpoint['state_dict']
    model.classifier = checkpoint['classifier']
    #class_to_idx = checkpoint['class_to_idx']
    #state_dict = checkpoint['state_dict']
    
    #model = getattr(model, class_to_idx)
    #model = getattr(model.classifier.load_state_dict, state_dict)
    
    print("model:", model)
    
    return model

def crop_image(image, new_size):
    width, height = image.size  
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2

    return image.crop((left, top, right, bottom))


def process_image(image):
    print("processing image")
    size = 256, 256
    pil_image = Image.open(image)
    pil_image.resize(size)
    
    pil_image = crop_image(pil_image,224)
    print(pil_image)
   
    np_image = np.array(pil_image)
    
    np_image = (np_image/255)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    
    print("image has been processed")
    return np_image


def predict(image_path, model, top_k, device):
    #Predict the class (or classes) of an image using a trained deep learning model.
    
    print("run predict function")
    # TODO: Implement the code to predict the class from an image file
    
    #Put the model in the correct mode 
    model.eval() 
    
    model = model.to(device) 

    # Process the image 
    img = process_image(image_path) 
    print('img:',img)

    conv_dim = torch.tensor(np.array([img])).type(torch.FloatTensor)
    print('conv_dim:',conv_dim)
    with torch.no_grad(): 
        
        output = model.forward(conv_dim)
        print('output orig:', output)
        output = torch.exp(output)
        print('output new:',output)

    #class_to_idx = model.class_to_idx 
    prob, index = output.topk(top_k)
    print("output.topk" ,output.topk)
    print("top_k" , top_k)
    print("prob" ,prob)
    print("index" ,index)
    
    topk_class = index.cpu().numpy()[0]

    print("topk_class" , topk_class)
    idx_to_classes = {v:k for k,v in model.class_to_idx.items()}
    print("idx_to_classes" ,idx_to_classes)
    classes = [idx_to_classes[idx] for idx in topk_class]  
    print("classes" ,classes)
        
    # Invert class_to_idx 
    #idx_to_class = {value: key for key, value in class_to_idx.items()} 
    #top_k = args.top_k

    index_topk = []
    prob_topk = []
    
    print(top_k)
    
    for i in range (0,top_k):
        index_topk.append(index[0][i])
        prob_topk.append(prob[0][i])

    index_topk = np.array(index_topk) 
    prob_topk = np.array(prob_topk) 
    
    print(index_topk)
    print(prob_topk)
    
    return (prob_topk, index_topk)


'''
def predict(image_path, model, topk, device):
    # Predict the class (or classes) of an image using a trained deep learning model.
    
       
    model.eval() 
    device = 'cpu'

    # Use the gpu if available 

    model = model.to(device) 

    # Process the image 
    
    img = process_image(image_path) 

    conv_dim = torch.tensor(np.array([img])).type(torch.FloatTensor) 
    with torch.no_grad(): 
        output = model.forward(conv_dim) 
        output = torch.exp(output) 

    # Assign class_to_idx 

    class_to_idx = model.class_to_idx 

    # Invert class_to_idx 

    idx_to_class = {value: key for key, value in class_to_idx.items()} 
    
    prob, index = output.topk(5) 

    index_top5 = np.array([index[0][0], index[0][1],index[0][2], index[0][3],index[0][4]]) 
    prob_top5 = np.array([prob[0][0], prob[0][1],prob[0][2], prob[0][3],prob[0][4]]) 
    
    return (prob_top5, index_top5)

'''

def main():
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    args = args_parser()
    print(args)
    print("GPU:" , args.gpu)
    #is_gpu=args.gpu
    
    if(args.gpu == True):
        device = "cuda"
    else:
        device = "cpu"
    
    model_checkpoint = load_checkpoint(args.checkpoint)
    
    print("!!!!!!!!!!")
    print(model_checkpoint)
    print("!!!!!!!!!!")
    probs, classes = predict(args.img, model_checkpoint, args.top_k,device)
    
    with open(args.names_file, 'r') as f:
        cat_to_name = json.load(f)
    
    print('Top probabilties:', probs)
    
    named_classes = []  
    for name in classes:
        print("name the flowers")
        named_classes.append(cat_to_name[str(name)])
    
    print('Top classes:', named_classes)
    
    
if __name__ == '__main__': main()
