import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import numpy as np
import json
import re
# loacl import
import sys
sys.path.append('../')
from config import preprocess
from utils import initialize_model

config_path = '/tf/orange/adversarial-attacks-pytorch/demos/config/model_config.json'

class TORCH_MODEL:
    
    def __init__(self, model_name='RESNET50', dataset='cifar10', gpu = False):
        
        if gpu == True:
            map_location = 'cuda:0'
        else:
            map_location = 'cpu'
        
        self.config = json.load(open(config_path))['torch_{}_{}'.format(model_name,dataset)]
        self.model_name = self.config['model']
        self.dataset = self.config['dataset']
        self.model_input_shape = self.config['model_input_shape'][0]
        self.preprocess = self.config['preprocess']
        
        # load model
        model_path = '/tf/models/{}_{}.pth'.format(self.model_name, dataset)        
        
        types = types = re.findall("^([a-z]+)", self.model_name)[0]
        model, input_size = initialize_model(types, self.config['label'], False, use_pretrained=True)
        model.load_state_dict(torch.load(model_path, map_location = map_location)['net'])
        
        self.model = model

    def predict(self, imgs):
        
        ### handle 3 dim image
        imgs = np.array(imgs).astype(np.uint8)
        if imgs.ndim == 3:
            imgs = [imgs]
            
        xs = []
        for img in imgs:
            ### check img before transform, the input img should be between [0,255] and type int
            if img.max()>1.0:
                img = img / 255.0
            
            ### preprocess the image
            resize = transforms.Resize(self.model_input_shape)
            im = getattr(preprocess, self.preprocess)(img)
            im = resize(im)
            xs.append(im.float())
            
        inputs = torch.stack(xs)

        ### set model to evaluate mmodel
        self.model.eval()

        ### put image to the model
        with torch.no_grad():
            outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        return outputs.numpy(), predicted.numpy()
    
    def zero_grad(self):

        self.model.zero_grad()
