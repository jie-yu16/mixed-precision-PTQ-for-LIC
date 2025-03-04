# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.ops import compute_padding
from compressai.zoo import image_models as pretrained_models
from compressai.zoo.image import model_architectures as architectures
from compressai.zoo.image_vbr import model_architectures as architectures_vbr

from .datasets.dataset import get_dataloader, get_calib_samples, get_calib_data

from .utils import setup_logger
from .sens_analysis import SummaryTools, draw_act_histogram, draw_act_channel,  draw_sensitive_influence
from .quant_operator import collect_stats, compute_amax, model_quant_disable, model_quant_enable, model_module_replace, model_bias_correction
import logging
from datetime import datetime
import random
import os
import numpy as np
from torchsummary import summary

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

architectures.update(architectures_vbr)
def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr-rgb"] = psnr(org, rec).item()
    metrics["ms-ssim-rgb"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)

                    
@torch.no_grad()
def inference(model, x, vbr_stage=None, vbr_scale=None):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2

    x_padded = F.pad(x, pad, mode="constant", value=0)
    
    start = time.time()
    out_enc = (
        model.compress(x_padded)
        if vbr_scale is None
        else model.compress(x_padded, stage=vbr_stage, s=0, inputscale=vbr_scale)
    )
    enc_time = time.time() - start

    start = time.time()
    out_dec = (
        model.decompress(out_enc["strings"], out_enc["shape"])
        if vbr_scale is None
        else model.decompress(
            out_enc["strings"],
            out_enc["shape"],
            stage=vbr_stage,
            s=0,
            inputscale=vbr_scale,
        )
    )
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, vbr_stage=None, vbr_scale=None):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2

    x_padded = F.pad(x, pad, mode="constant", value=0)

    start = time.time()
    out_net = (
        model.forward(x_padded)
        if vbr_scale is None
        else model.forward(x_padded, stage=vbr_stage, inputscale=vbr_scale)
    )
    elapsed_time = time.time() - start
    
    out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True, progress=False
    ).eval()


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]

    model_cls = architectures[arch]
    if arch in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]:
        net = model_cls.from_state_dict(state_dict, vr_entbttlnck=True)
        if not no_update:
            net.update(force=True, scale=net.Gain[-1])
    else:
        net = model_cls.from_state_dict(state_dict)
        if not no_update:
            net.update(force=True)

    return net.eval()


def eval_model(
    model: nn.Module,
    outputdir: Path,
    inputdir: Path,
    filepaths,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    vbr_stage=None,
    vbr_scale=None,
    **args: Any,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    is_vbr_model = args["architecture"].endswith("-vbr")
    for filepath in filepaths:
        x = read_image(filepath).to(device)
        if args["half"]:
            model = model.half()
            x = x.half()        
        if not entropy_estimation:
            rv = (
                inference(model, x)
                if not is_vbr_model
                else inference(model, x, vbr_stage, vbr_scale)
            )
        else:
            rv = (
                inference_entropy_estimation(model, x)
                if not is_vbr_model
                else inference_entropy_estimation(model, x, vbr_stage, vbr_scale)
            )
        for k, v in rv.items():
            metrics[k] += v
        if args["per_image"]:
            if not Path(outputdir).is_dir():
                raise FileNotFoundError("Please specify output directory")

            output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(
                inputdir
            )
            output_subdir.mkdir(parents=True, exist_ok=True)
            image_metrics_path = output_subdir / f"{filepath.stem}-{trained_net}.json"
            with image_metrics_path.open("wb") as f:
                output = {
                    "source": filepath.stem,
                    "name": args["architecture"],
                    "description": f"Inference ({description})",
                    "results": rv,
                }
                f.write(json.dumps(output, indent=2).encode())

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics
    
def build_sensitivity_profile(model, cali_data, save_file):
    
    device = next(model.parameters()).device
    summary =  SummaryTools(save_file)

    quant_layer_names = []
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
        #if name.endswith("_input_quantizer"): #onlyweight quantization
            module.disable()
            layer_name = name.replace("._input_quantizer", "").replace("._weight_quantizer", "")
            if layer_name not in quant_layer_names:
                quant_layer_names.append(layer_name)
    print(f"Sensitive analysis by each layer...")

    for i, quant_layer in enumerate(quant_layer_names):
        print("Enable", quant_layer)
        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
            #if name.endswith("_input_quantizer") and quant_layer in name:                
                module.enable()
                print(F"{name:40}: {module}")

        metrics = defaultdict(float)
        for data in cali_data:
            x = data.to(device)  
            out_net = model(x)
            metric_one = compute_metrics(x, out_net["x_hat"], 255)
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )            
            rv = {"psnr-rgb": metric_one["psnr-rgb"],"ms-ssim-rgb": metric_one["ms-ssim-rgb"],"bpp": bpp.item()}
            
            for k, v in rv.items():
                metrics[k] += v
                
        for k, v in metrics.items():
            metrics[k] = v / len(cali_data)                
        
        #dist_avg, distloss_avg, bit_avg
        summary.append([metrics['psnr-rgb'], metrics['ms-ssim-rgb'], metrics['bpp'], quant_layer])

        for name, module in model.named_modules():
            if name.endswith("_quantizer") and quant_layer in name:
            #if name.endswith("_input_quantizer") and quant_layer in name:   
                module.disable()
                print(F"{name:40}: {module}")
    
    summary = sorted(summary.data, key=lambda x: x[1], reverse=True)
    print("Sensitive Summary")

def setup_args():
    # Common options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parent_parser.add_argument(
        "-d",
        "--output_directory",
        type=str,
        default="",
        help="path of output directory. Optional, required for output json file, results per image. Default will just print the output results.",
    )
    parent_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )
    parent_parser.add_argument(
        "--per-image",
        action="store_true",
        help="store results for each image of the dataset, separately",
    )
    parent_parser.add_argument(
        "--lambdaall",
        nargs='+',
        type=float,
        default=[0.0018, 0.00035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800],
        help="lambda value for different bitrate model. ",
    )     
    # Options for variable bitrate (vbr) models
    
    parent_parser.add_argument(
        "--vbr_quantstep",
        dest="vbr_quantstepsizes",
        type=str,
        default="10.0000,7.1715,5.1832,3.7211,2.6833,1.9305,1.3897,1.0000",
        help="Quantization step sizes for variable bitrate (vbr) model. Floats [10.0 , 1.0] (example: 10.0,8.0,6.0,3.0,1.0)",
    )
    parent_parser.add_argument(
        "--vbr_tr_stage",
        type=int,
        choices=[1, 2],
        default=2,
        help="Stage in vbr model training. \
            1: Model behaves/runs like a regular single-rate \
            model without using any vbr tool (use for training/testing a model for single/highest lambda). \
            2: Model behaves/runs like a vbr model using vbr tools. (use for post training stage=1 result.)",
    )
    
    # general parameters for data and model
    parent_parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parent_parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size for data loader')
    parent_parser.add_argument('--worker_num', default=4, type=int, help='number of workers for data loader')
    parent_parser.add_argument('--data_path', default='./dataset', type=str, help='calib-data dir for data loader')
    parent_parser.add_argument('--c_data', default='clic41', type=str, help='calib-data for data loader')
    
    parent_parser.add_argument('--lmbda', default=0.0483, type=float, 
                        help='the lmbda related to quality, 0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483')
    parent_parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str, help='result dir name')
    
    # quantization parameters
    parent_parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parent_parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parent_parser.add_argument('--act_channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parent_parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parent_parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parent_parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parent_parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parent_parser.add_argument('--num_samples', default=10, type=int, help='size of the calibration dataset')
    parent_parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parent_parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parent_parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parent_parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parent_parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parent_parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parent_parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parent_parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parent_parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parent_parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    parent_parser.add_argument('--init', default='max', type=str, help='param init type', 
                        choices=['max','mse', 'gaussian', 'l1', 'l2', ])
                        
    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        type=str,
        default="1",
        help="Pretrained model qualities. (example: '1,2,3,4') (default: %(default)s)",
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="checkpoint_paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )
    return parser

            
def main(argv):  # noqa: C901
    parser = setup_args()
    args = parser.parse_args(argv)
    seed_all(args.seed)
    
    quant_desc_input = QuantDescriptor(num_bits=8, calib_method="max")#  #histogram, max, calib_method="histogram", axis=(1)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)

    quant_desc_weight = QuantDescriptor(num_bits=8, axis=(0)) #histogram, max
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)        
    
#    quant_modules.initialize() 
     
    if args.source not in ["checkpoint", "pretrained"]:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    is_vbr_model = args.architecture.endswith("-vbr")

    # create output directory
    if args.output_directory:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    if args.source == "pretrained":
        args.qualities = [int(q) for q in args.qualities.split(",") if q]
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        if is_vbr_model:
            opts += (0,)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d} "
    else:
        runs = args.checkpoint_paths
        opts = (args.architecture, args.no_update)
        if is_vbr_model:
            opts += (args.checkpoint_paths[0],)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s} "

    if is_vbr_model:
        if args.source == "checkpoint":
            assert (
                len(args.checkpoint_paths) <= 1
            ), "Use only one checkpoint for vbr model."
        scales = [1.0 / float(q) for q in args.vbr_quantstepsizes.split(",") if q]
        runs = sorted(scales)
        runs = torch.tensor(runs)
        log_fmt = "\rEvaluating quant step {run:5.2f} "
        model = load_func(*opts)
        # set some arch specific params for vbr
        model.no_quantoffset = False
        if args.architecture in ["mbt2018-vbr"]:
            model.scl2ctx = True
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            runs = runs.to("cuda")

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(
                log_fmt.format(*opts, run=(run if not is_vbr_model else 1.0 / run))
            )
            sys.stderr.flush()
        if not is_vbr_model:
            model = load_func(*opts, run)
        else:
            # update bottleneck for every new quant_step if vbr bottleneck is used in the model
            if (
                args.architecture in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]
                and args.vbr_tr_stage == 2
            ):
                model.update(force=True, scale=run)
        if args.source == "pretrained":
            trained_net = f"{args.architecture}-{args.metric}-{run}-{description}"
        else:
            run_ = run if not is_vbr_model else args.checkpoint_paths[0]
            cpt_name = Path(run_).name[: -len(".tar.pth")]  # removesuffix() python3.9
            trained_net = f"{cpt_name}-{description}"
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available() and not is_vbr_model:
            model = model.to("cuda")
        
        model.eval()
        ########## Begin Quantization and Calibration #######################
        setup_logger(args, run)
        
        for name, child in model.named_children():
            logging.info(f"{name}: {child}")
                     
        #normalization same size
        #train_dataloader = get_dataloader(args.data_path, args.c_data, args.batch_size, args.worker_num)
        #cali_data = get_calib_samples(train_dataloader, num_samples=args.num_samples)
        
        cali_data = get_calib_data(filepaths, args.num_samples)
        
        sens_dir = f"./results/sens_analysis/{args.architecture}/{run}/"
        os.makedirs(sens_dir, exist_ok=True)
                   
        with open(sens_dir + 'layer_name_2.txt', 'r') as file:
            layer_names = file.read().splitlines()
        
        with open(f"./results/sens_analysis/{args.architecture}/" + 'layer_name_relu.txt', 'r') as file:
            relu_layer = file.read().splitlines()

        nosens_layer = layer_names[6:]
            
        sens_layer = layer_names[:6]

        
#        sens_layer_relu = []
#        sens_layer_norelu = []
#        nosens_layer_relu = []
#        nosens_layer_norelu = []
#        for name in sens_layer[:]:
#            if name in relu_layer:
#                sens_layer_relu.append(name)
#            else:
#                sens_layer_norelu.append(name)                 
#        
#        for name in nosens_layer[:]:
#            if name in relu_layer:
#                nosens_layer_relu.append(name)
#            else:
#                nosens_layer_norelu.append(name) 
#                
#        sens_quant_desc_input_relu = QuantDescriptor(num_bits=8, unsigned = True, calib_method="max")    #
#        sens_quant_desc_weight_relu = QuantDescriptor(num_bits=8, axis=(0))  
#        sens_quant_desc_input_norelu = QuantDescriptor(num_bits=8, calib_method="max")#, calib_method="histogram", calib_method="max", axis=(1)
#        sens_quant_desc_weight_norelu = QuantDescriptor(num_bits=8, axis=(0)) 
#        
#        nosens_quant_desc_input_relu = QuantDescriptor(num_bits=8, unsigned = True, calib_method="max")#, unsigned = True
#        nosens_quant_desc_weight_relu = QuantDescriptor(num_bits=8, axis=(0))
#        nosens_quant_desc_input_norelu = QuantDescriptor(num_bits=8, calib_method="max")
#        nosens_quant_desc_weight_norelu = QuantDescriptor(num_bits=8, axis=(0)) 
#        
#        model_module_replace(model, sens_layer_relu, sens_quant_desc_input_relu, sens_quant_desc_weight_relu)
#        model_module_replace(model, sens_layer_norelu, sens_quant_desc_input_norelu, sens_quant_desc_weight_norelu)
#        model_module_replace(model, nosens_layer_relu, nosens_quant_desc_input_relu, nosens_quant_desc_weight_relu)
#        model_module_replace(model, nosens_layer_norelu, nosens_quant_desc_input_norelu, nosens_quant_desc_weight_norelu)

      
#        for name, param in model.named_parameters():
#        
#            if name.endswith(".weight") or name.endswith(".bias"):
#                
#                layer_name = name.replace(".weight", "").replace(".bias", "")
#                if layer_name == "g_s.0":
#                    param.data.div_(1/1.5)
#                if layer_name == "g_s.2":
#                    param.data.div_(1.5)  
#                if layer_name == "g_s.4":
#                    param.data.div_(1/1.5)
#                if layer_name == "g_s.6":
#                    param.data.div_(1.5)                         
#            if name == 'g_s.1.gamma' or name == 'g_s.5.gamma':
#                with torch.no_grad():
#                    param.data.div_(1.5) 
                                  
#        with torch.no_grad():
#            collect_stats(model, cali_data) #, num_batches=1,args.dataset, args.num_samples
#            compute_amax(model, method="percentile", percentile=99.9999)    # entropy/method="percentile", percentile=99.9, mse, entropy             
#            
            
            # model_bias_correction(model, cali_data)
            # draw_act_histogram(model, sens_dir)   
            # draw_act_channel(model, sens_dir)
                               
        # summary(model, input_size=(3, 256, 256))
        
        device = next(model.parameters()).device
        
        sens_file = sens_dir + f"sens.json"
#        
#        build_sensitivity_profile(model, cali_data, sens_file)
#        draw_sensitive_influence(sens_dir, sens_file, mode = 0)        
#        draw_sensitive_influence(sens_dir, sens_file, mode = 1)
#        draw_sensitive_influence(sens_dir, sens_file, mode = 2, lamda = args.lambdaall[run-1]*(255**2)) 

#        model_quant_enable(model)  
#        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        class EncodeWrapper(nn.Module):
          def __init__(self, model):
              super(EncodeWrapper, self).__init__()
              self.model = model
      
          def forward(self, dummy_input1):
              return self.model.encode(dummy_input1)
        
        model_dir = f"./results/save_model/{args.architecture}/{run}/"
        os.makedirs(model_dir, exist_ok=True)

        dummy_input1 = torch.randn(size=(1, 3, 2560, 2560), dtype=torch.float32, device=device)
        wrapped_model = EncodeWrapper(model)
        torch.onnx.export(
            wrapped_model,
            dummy_input1,
            model_dir + "iframe_encode_fp32_2560_2560.onnx",
            input_names=["input.x"],
            output_names=["output.y", "output.z"],
            verbose=True,
            opset_version=16,
            # dynamic_axes={
            #                 "input.x": {0: 'batch_size', 2: 'input_height', 3: 'input_width'},
            #                 "output.x_rec": {0: 'batch_size', 2: 'output_height', 3: 'output_width'},
            #                 "output.img_rate":{0: 'batch_size'}
            #                 }
        )
        
#        dummy_input1 = torch.randn(size=(1, 3, 1280, 1280), dtype=torch.float32, device=device) #1280, 2560
#        torch.onnx.export(
#            model,
#            dummy_input1,
#            model_dir + "iframe_fp32noquanti_1280_1280.onnx",
#            input_names=["input.x"],
#            output_names=["output.x_rec","output.y_likelihoods", "output.z_likelihoods"],
#            verbose=True,
#            opset_version=16,
#            # dynamic_axes={
#            #                 "input.x": {0: 'batch_size', 2: 'input_height', 3: 'input_width'},
#            #                 "output.x_rec": {0: 'batch_size', 2: 'output_height', 3: 'output_width'},
#            #                 "output.img_rate":{0: 'batch_size'}
#            #                 }
#        )        
#                        
#        torch.save(model.state_dict(), model_dir +"iframe-calibrated-int8.pth")       
              
  
        args_dict = vars(args)
        metrics = eval_model(
            model,
            args.output_directory,
            args.dataset,
            filepaths,
            trained_net=trained_net,
            description=description,
            vbr_stage=None if not is_vbr_model else args.vbr_tr_stage,
            vbr_scale=None if not is_vbr_model else run,
            **args_dict,
        )
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        f"entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": f"{args.architecture}-tensorrt-pertensor",
        "description": f"Inference ({description})",
        "results": results,
    }
    if args.output_directory:
        output_file = (
            args.output_file
            if args.output_file
            else f"{output['name']}"
        )

        with (Path(f"{args.output_directory}/{output_file}").with_suffix(".json")).open(
            "wb"
        ) as f:
            f.write(json.dumps(output, indent=2).encode())

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
