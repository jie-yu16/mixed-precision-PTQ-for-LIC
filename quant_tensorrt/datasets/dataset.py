import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from compressai.ops import compute_padding
from PIL import Image
import torch.nn.functional as F

def get_dataloader(data_path, c_data, batchsize, worker_num):
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(1300, 2000), transforms.ToTensor()]
    )
    train_dataset = ImageFolder(data_path, split=c_data, transform=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        num_workers=worker_num,
        shuffle=True,
        pin_memory=True,
    )

    '''
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(256), transforms.ToTensor()]
    )
    test_dataset = ImageFolder(config['trainset'], split=config['t_data'], transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batchsize_test'],
        num_workers=config['worker_num'],
        shuffle=False,
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader
    '''
    return train_dataloader
    
'''
def get_train_samples_imagenet(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples] # output 3-dim tensor
'''

def get_calib_samples(train_loader, num_samples):
    train_data = None
    for i, batch in enumerate(train_loader):
        if train_data is None:
            train_data = batch
        else:
            train_data = torch.cat((train_data, batch))
        
        if train_data.shape[0] >= num_samples:
            break
    return train_data[:num_samples] # output 4-dim tensor for LIC
    
def get_calib_data(filepaths, num_samples):

    calib_data = []
    for i, filepath in enumerate(filepaths):
        img = Image.open(filepath).convert("RGB")
        x = transforms.ToTensor()(img)
        x = x.unsqueeze(0)
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(x, pad, mode="constant", value=0)
                
        calib_data.append(x_padded)
        
        if i >= num_samples-1:
            break
            
    return calib_data    

            
