import os
import sys
import yaml
import time
import struct
import shutil

import logging
from PIL import Image
from shutil import copy2
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor

import json
import numpy as np
import matplotlib.pyplot as plt
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d
from pytorch_quantization.nn.modules.quant_linear import QuantLinear

def collect_stats(model, cali_data): #, num_batches
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    
    device = next(model.parameters()).device
    
    for i,data in tqdm(enumerate(cali_data)):
        with torch.no_grad():
            _,_ = model(data.to(device))

                     
    '''
    for i in range(int(cali_data.size(0) / batch_size)): # , total=num_batches
        data = cali_data[i * batch_size:(i + 1) * batch_size]
        model(data.cuda())
    '''       
    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                    
                    
def model_quant_disable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()
            
def model_quant_enable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()
            module.enable_quant()
            module.disable_calib()
            
def model_module_replace(model, sens_layer, quant_desc_input_sens, quant_desc_weight_sens):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer) and name.replace("._input_quantizer", "") in sens_layer:
            new_module = quant_nn.TensorQuantizer(quant_desc_input_sens)
            parent_name = name.rsplit('.', 1)[0]
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, name.split('.')[-1], new_module)
            else:
                setattr(model, name, new_module)  
       
        if isinstance(module, quant_nn.TensorQuantizer) and name.replace("._weight_quantizer", "") in sens_layer:
            new_module = quant_nn.TensorQuantizer(quant_desc_weight_sens)
            parent_name = name.rsplit('.', 1)[0]
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, name.split('.')[-1], new_module)
            else:
                setattr(model, name, new_module)  

def model_bias_correction(model, cali_data):          
    device = next(model.parameters()).device
    bias_all = {}
    
    pic_num = len(cali_data)

    for i,data in tqdm(enumerate(cali_data)):
        model_quant_disable(model)   
    
        with torch.no_grad():
            _,_ = model(data.to(device))
         
        for name, module in model.named_modules():
            if isinstance(module, (QuantConv2d, QuantLinear, QuantConvTranspose2d)):          
                for sub_name, sub_module in module._modules.items():
                    if isinstance(sub_module, quant_nn.TensorQuantizer) and sub_name.endswith("_input_quantizer"):
                        org_data = sub_module._data
                
                output = module(org_data)
                
                for sub_name, sub_module in module._modules.items():
                    if isinstance(sub_module, quant_nn.TensorQuantizer):
                        sub_module.enable()
                        sub_module.enable_quant()
                        sub_module.disable_calib()
                
                output_quant = module(org_data)   
                
                output_diff =  output_quant - output 
                        
                mean_bias = torch.mean(output_diff, dim=(0,2,3))
                if name not in bias_all:
                     bias_all[name] = mean_bias
                else:
                     bias_all[name] = bias_all[name] + mean_bias
                                                  
    
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            layer_name = name.replace(".bias", "")
            if layer_name in bias_all:
                #print(param.data)#, bias_all[layer_name].shape)
                param.data = param.data - bias_all[layer_name]/pic_num  