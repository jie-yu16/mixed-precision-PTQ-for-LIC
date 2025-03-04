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

def setup_logger(args, run):
    log_dir = f'./results/logs/{args.architecture}-{args.metric}/tensorrt/{run}/'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = log_dir + time.strftime('%Y%m%d_%H%M%S') + '.log'
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_file, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_file)
    
             
            

                
