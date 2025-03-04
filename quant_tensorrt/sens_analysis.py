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

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor

import json
import numpy as np
import matplotlib.pyplot as plt
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib


def draw_sensitive_influence(sens_dir, sens_file, mode = 0, lamda = 128):

    folder_path = sens_dir
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(sens_file, "r") as file:
        data = json.load(file)
    
    if mode == 0:
        data = sorted(data, key=lambda x: x[0], reverse=False)
    elif mode == 1:
        data = sorted(data, key=lambda x: x[2], reverse=True)    
    else:
        data = sorted(data, key=lambda x: lamda / (10 ** (x[0] / 10)) + x[2], reverse=True)
    
    dist_list = []
    rate = []
    layer_name = []
    for i in range(len(data)):
        dist_list.append(data[i][0])#[0]
        layer_name.append(data[i][3])
        rate.append(data[i][2])
            
    file_path = os.path.join(folder_path, f"PSNR_{mode}.txt")
    np.savetxt(file_path, np.vstack(dist_list), fmt='%f')

    file_path = os.path.join(folder_path, f"layer_name_{mode}.txt")
    with open(file_path, 'w') as file:
        for item in layer_name:
            file.write(f"{item}\n")

    file_path = os.path.join(folder_path, f"rate_{mode}.txt")
    np.savetxt(file_path, rate, fmt='%f')

    plt.plot(dist_list, label='PSNR_RGB')

    plt.title("The influence of different sensitive layer to PSNR")
    plt.xlabel("layer")
    plt.ylabel("PSNR")
    plt.legend(loc='upper right')
    file_path = os.path.join(folder_path, f"sensitive_psnr_{mode}.png")
    plt.savefig(file_path)
    plt.close()

    plt.plot(rate, label='rate/bpp')
    plt.title("The influence of different sensitive layer to rate/bpp")
    plt.xlabel("layer")
    plt.ylabel("rate/bpp")
    plt.legend(loc='upper right')
    file_path = os.path.join(folder_path, f"sensitive_bpp_{mode}.png")
    plt.savefig(file_path)
    plt.savefig(file_path)
    plt.close()   


def replace_to_quantization_model(model, data_dir, ignore_layer_num = 0, mode = 0):
    """replace_to_quantization_model
    This function replace some sensitive layers from data_dir and disable quanization
    
    Args:
        model: pytorch model
        data_dir: sensitive layers file
        mode: 0(bpp) 1(distortion)
    """
    with open(data_dir, "r") as file:
        data = json.load(file)
    
    if mode == 0:
        data = sorted(data, key=lambda x: x[2], reverse=True)
    elif mode == 1:
        data = sorted(data, key=lambda x: x[1], reverse=True)
    else:
        data = sorted(data, key=lambda x: 2048*x[1] + x[2], reverse=True)

    quant_layer_names = []

    for i in range(ignore_layer_num):
        quant_layer_names.append(data[i][3])

    for name, module in model.named_modules():
        if name.endswith("_quantizer"):  
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name in quant_layer_names:
                module.disable() 
                print(F"Dsiable {name:40}: {module}")



class SummaryTools:

    def __init__(self, file):
        self.file = file
        self.data = []
    
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def draw_act_histogram(model, folder_path):
    # Load calib result
    folder_path = folder_path + "act_histogram/"
    if not os.path.exists(folder_path):
       os.makedirs(folder_path)
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if isinstance(module._calibrator, calib.HistogramCalibrator):
                hist = module._calibrator._calib_hist
                bin_edges = module._calibrator._calib_bin_edges
                clib_amax = module._calibrator.compute_amax('mse', stride=1, start_bin=128, percentile=99.99)
                clib_amax1 = module._calibrator.compute_amax('percentile', stride=1, start_bin=128, percentile=99.99)
                clib_amax2 = module._calibrator.compute_amax('percentile', stride=1, start_bin=128, percentile=99.999)
                clib_amax3 = module._calibrator.compute_amax('percentile', stride=1, start_bin=128, percentile=99.9999)
                clib_amax4 = module._calibrator.compute_amax('entropy', stride=1, start_bin=128, percentile=99.9999)
                
                x_coords = [bin_edges[-1], clib_amax.cpu(), clib_amax1.cpu(), clib_amax2.cpu(), clib_amax3.cpu(), clib_amax4.cpu()]
                titles = ["max", "mse", "99.99%", "99.999%", "99.9999%", "entropy"]
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
                
                for x, color in zip(x_coords, colors):
                   plt.axvline(x, color = color, linestyle='--')
                   
                handles = [plt.Line2D([0], [0], color=color, linestyle='--', label=f'{title} ({x})') for x, color, title in zip(x_coords, colors, titles)]
                plt.legend(handles=handles, loc='upper right')
                plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='teal')
                plt.title(f'Histogram of Layer {name} input')
                plt.yscale('log')
                # plt.xlim(left=0)
                
                file_path = os.path.join(folder_path, f'{name}.png')
                plt.savefig(file_path)
                plt.close()


def draw_act_channel(model, folder_path):
    # Load calib result
    folder_path = folder_path + "act_channel/"
    if not os.path.exists(folder_path):
       os.makedirs(folder_path)
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if isinstance(module._calibrator, calib.MaxCalibrator):
                amax = module._amax.view(-1).cpu()
                
                channels = range(len(amax))
                
                plt.figure(figsize=(10, 5))
                plt.vlines(channels, ymin=0, ymax=amax.numpy(), color='b', alpha=0.7) 
                plt.xlabel('Channel')
                plt.ylabel('Range Value')
                plt.title(f'Ranges of different channels of Layer {name}')
                file_path = os.path.join(folder_path, f'{name}.png')
                plt.savefig(file_path)
                plt.close()