
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockGoDown(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, n_layers, activation_class=nn.LeakyReLU , dropout_p=0.3, reverse=False, do_scale=True, change_size=0):
        super(ResidualBlockGoDown, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.activation_class = activation_class
        self.dropout_p = dropout_p
        self.reverse = reverse
        self.do_scale = do_scale
        self.change_size = change_size

        # https://proproprogs.ru/nn_pytorch/pytorch-klassy-sequential-modulelist-i-moduledict
        # http://arxiv.org/pdf/1502.03167   -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        residual_go_down_start = nn.Sequential()
        residual_go_down_start.add_module("Conv2d", nn.Conv2d(in_channels = self.c_in, out_channels = self.c_out,
                                                              kernel_size = self.kernel_size, padding = (self.kernel_size - 1) // 2, bias=False))
        residual_go_down_start.add_module("BatchNorm2d", nn.BatchNorm2d(self.c_out))
        residual_go_down_start.add_module("activation", self.activation_class())
        residual_go_down_start.add_module("Dropout2d", nn.Dropout2d(self.dropout_p))

        residual_go_down = nn.Sequential()
        residual_go_down.add_module("Conv2d", nn.Conv2d(in_channels = self.c_out, out_channels = self.c_out,
                                                        kernel_size = self.kernel_size, padding = (self.kernel_size - 1) // 2, bias=False))
        residual_go_down.add_module("BatchNorm2d", nn.BatchNorm2d(self.c_out))
        residual_go_down.add_module("activation", self.activation_class())
        residual_go_down.add_module("Dropout2d", nn.Dropout2d(self.dropout_p))

        self.blocks_go_down = nn.Sequential()
        self.blocks_go_down.add_module("block_0", residual_go_down_start)
        for layer_ind in range(1, n_layers):
            self.blocks_go_down.add_module(f"block_{layer_ind}", residual_go_down)
        ############################################################################

        self.math_with_previous = nn.Conv2d(self.c_in, self.c_out, 1) if self.c_in != self.c_out else nn.Identity()


        ## Pooling ##
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        if self.do_scale:
            return self.pooling(self.math_with_previous(x) + self.blocks_go_down(x))
        return self.math_with_previous(x) + self.blocks_go_down(x)


class Decoder(nn.Module):
    def __init__(self, num_blocks, block_sizes, hidden_sizes, kernel_sizes, activation_classes, use_scales, change_sizes, dropout_ps):
        super(Decoder, self).__init__()

        def fill_list(val, size):
            if isinstance(val, list):
                assert len(val) == size
                return val
            return [val for _ in range(size)]

        self.num_blocks = num_blocks
        self.block_sizes = block_sizes
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.activation_classes = activation_classes
        self.use_scales = use_scales
        self.change_sizes = change_sizes
        self.dropout_ps = dropout_ps

        self.generator = nn.Sequential()
        for block_ind in range(self.num_blocks):
            self.generator.add_module(f"residual_{block_ind}", ResidualBlockGoDown( c_in=self.hidden_sizes[block_ind],
                                                                                          c_out=self.hidden_sizes[block_ind+1],
                                                                                          kernel_size=self.kernel_sizes[block_ind],
                                                                                          n_layers=self.block_sizes[block_ind],
                                                                                          activation_class=self.activation_classes[block_ind],
                                                                                          dropout_p=self.dropout_ps[block_ind],
                                                                                          reverse=True,
                                                                                          do_scale=self.use_scales[block_ind],
                                                                                          change_size=self.change_sizes[block_ind])
                                      )


    def forward(self, x):
        return self.generator(x)



class ResidualBlockGoUp(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, n_layers, activation_class=nn.LeakyReLU , dropout_p=0.3, reverse=False, do_scale=True, change_size=0):
        super(ResidualBlockGoUp, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.activation_class = activation_class
        self.dropout_p = dropout_p
        self.reverse = reverse
        self.do_scale = do_scale
        self.change_size = change_size

        # https://proproprogs.ru/nn_pytorch/pytorch-klassy-sequential-modulelist-i-moduledict
        # http://arxiv.org/pdf/1502.03167   -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        residual_go_up_start = nn.Sequential()
        residual_go_up_start.add_module("Conv2d", nn.ConvTranspose2d(in_channels = self.c_in, out_channels = self.c_out,
                                                                     kernel_size = self.kernel_size, padding = (self.kernel_size - 1) // 2, bias=False))
        residual_go_up_start.add_module("BatchNorm2d", nn.BatchNorm2d(self.c_out))
        residual_go_up_start.add_module("activation", self.activation_class())
        residual_go_up_start.add_module("Dropout2d", nn.Dropout2d(self.dropout_p))

        residual_go_up = nn.Sequential()
        residual_go_up.add_module("Conv2d", nn.ConvTranspose2d(in_channels = self.c_out, out_channels = self.c_out,
                                                               kernel_size = self.kernel_size, padding = (self.kernel_size - 1) // 2, bias=False))
        residual_go_up.add_module("BatchNorm2d", nn.BatchNorm2d(self.c_out))
        residual_go_up.add_module("activation", self.activation_class())
        residual_go_up.add_module("Dropout2d", nn.Dropout2d(self.dropout_p))

        self.blocks_go_up = nn.Sequential()
        self.blocks_go_up.add_module("block_0", residual_go_up_start)
        for layer_ind in range(1, n_layers):
            self.blocks_go_up.add_module(f"block_{layer_ind}", residual_go_up)
        ############################################################################

        self.math_with_previous = nn.Conv2d(self.c_in, self.c_out, 1) if self.c_in != self.c_out else nn.Identity()


        ## Pooling ##
        self.pooling = nn.ConvTranspose2d(in_channels=self.c_out, out_channels=self.c_out, stride=2, kernel_size=2)

    def forward(self, x):
        if self.do_scale:
            return self.pooling(self.math_with_previous(x) + self.blocks_go_up(x))
        return self.math_with_previous(x) + self.blocks_go_up(x)


class Encoder(nn.Module):
    def __init__(self, num_blocks, block_sizes, hidden_sizes, kernel_sizes, activation_classes, use_scales, change_sizes, dropout_ps, fc_size=0):
        super(Encoder, self).__init__()

        self.num_blocks = num_blocks
        self.block_sizes = block_sizes                
        self.hidden_sizes = hidden_sizes              
        self.kernel_sizes = kernel_sizes              
        self.activation_classes = activation_classes  
        self.use_scales = use_scales                  
        self.change_sizes = change_sizes              
        self.dropout_ps = dropout_ps                  
        self.fc_size = fc_size


        self.residual_blocks = nn.Sequential()
        for block_ind in range(self.num_blocks):
            self.residual_blocks.add_module(f"residual_{block_ind}", ResidualBlockGoUp( c_in=self.hidden_sizes[block_ind],
                                                                                        c_out=self.hidden_sizes[block_ind+1],
                                                                                        kernel_size=self.kernel_sizes[block_ind],
                                                                                        n_layers=self.block_sizes[block_ind],
                                                                                        activation_class=self.activation_classes[block_ind],
                                                                                        dropout_p=self.dropout_ps[block_ind],
                                                                                        reverse=False,
                                                                                        do_scale=self.use_scales[block_ind],
                                                                                        change_size=self.change_sizes[block_ind])
                                            )


        # prepair to classifier part (linear layer)
        self.vect = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # discriminator's verdict
        self.verdict = nn.Linear(self.hidden_sizes[-1], 2)


    def forward(self, x):
        x = self.residual_blocks(x)
        x = self.vect(x).squeeze(-1).squeeze(-1)
        return self.verdict(x)
    

class CycleGAN(nn.Module):
    def __init__(self, gen_a_b_params, gen_b_a_params, discr_a_params, discr_b_params):
        super(CycleGAN, self).__init__()

        self.generator_a_b = Decoder(**gen_a_b_params)
        self.generator_b_a = Decoder(**gen_b_a_params)
        self.discriminator_a = Encoder(**discr_a_params)
        self.discriminator_b = Encoder(**discr_b_params)

    def generate_a_b(self, x):
        return self.generator_a_b(x)

    def generate_b_a(self, x):
        return self.generator_b_a(x)


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms as tr
from torchvision.transforms import v2

import os
import sys
import requests
import cv2

from dataclasses import dataclass

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_transforms(mean, std, **hyperparams):
    train_transform = tr.Compose([
        v2.ToImage(),
        v2.Resize(hyperparams["resize"], antialias=True),
        v2.RandomHorizontalFlip(),
        v2.GaussianBlur(kernel_size=hyperparams["kernel_size"], sigma=hyperparams["sigma"]),
        v2.ColorJitter(
            brightness = hyperparams["brightness"],
            contrast = hyperparams["contrast"],
            saturation = hyperparams["saturation"],
        ),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    val_transform = tr.Compose([
        v2.ToImage(),
        v2.Resize(hyperparams["resize"], antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    def de_normalize(img, normalized=True, mean=mean, std=std):
        #print(img.shape)
        img = img.squeeze(0)
        img = img.detach().cpu().numpy().transpose((1, 2, 0))
        if normalized:
            return img * std + mean
        else:
            return img

    return train_transform, val_transform, de_normalize


def Inference_a(model, img_a, de_norm_b):
    model.eval()
    num_images = 1
    with torch.no_grad():
        img_a = img_a.to(device).unsqueeze(0)
    return de_norm_b(model.generator_a_b(img_a))#[0]

def Inference_b(model, img_b, de_norm_a):
    model.eval()
    num_images = 1
    with torch.no_grad():
        img_a = img_b.to(device).unsqueeze(0)
    return de_norm_a(model.generator_b_a(img_b))#[0]

from collections import defaultdict
from termcolor import colored


def beautiful_int(i):
    i = str(i)
    return ".".join(reversed([i[max(j, 0):j+3] for j in range(len(i) - 3, -3, -3)]))


def model_num_params(model, verbose_all=True, verbose_only_learnable=False):
    sum_params = 0
    sum_learnable_params = 0
    submodules = defaultdict(lambda : [0, 0])
    for name, param in model.named_parameters():
        num_params = param.numel()
        if verbose_all or (verbose_only_learnable and param.requires_grad):
            print(
                colored(
                    '{: <65} ~  {: <9} params ~ grad: {}'.format(
                        name,
                        beautiful_int(num_params),
                        param.requires_grad,
                    ),
                    {True: "green", False: "red"}[param.requires_grad],
                )
            )
        sum_params += num_params
        sm = name.split(".")[0]
        submodules[sm][0] += num_params
        if param.requires_grad:
            sum_learnable_params += num_params
            submodules[sm][1] += num_params

    return sum_params, sum_learnable_params

def create_model_and_optimizer(model_class, model_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, device=device):
    model = model_class(**model_params)
    model = model.to(device)

    optimizer_d = torch.optim.Adam(
        (
            [param for param in model.discriminator_a.parameters()]
            + [param for param in model.discriminator_b.parameters()]
        ),
        lr,
        betas=betas,
        eps=eps,
    )
    optimizer_g = torch.optim.Adam(
        (
            [param for param in model.generator_a_b.parameters()]
            + [param for param in model.generator_b_a.parameters()]
        ),
        lr,
        betas=betas,
        eps=eps,
    )
    return model, optimizer_d, optimizer_g


decoder_params = dict(
    num_blocks = 5,
    block_sizes = [3, 3, 3, 3, 3],
    hidden_sizes = [3, 16, 32, 32, 16, 3],
    kernel_sizes = [3]*5,
    activation_classes = [nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU,  nn.LeakyReLU, nn.Tanh],
    use_scales = [False, False, False, False, False],
    change_sizes = [0]*5,
    dropout_ps = [0.]*5,
)
encoder_params = dict(
    num_blocks = 3,
    block_sizes = [2, 2, 2],
    hidden_sizes = [3, 16, 32, 64],
    kernel_sizes = [3]*3,
    activation_classes = [nn.LeakyReLU,  nn.LeakyReLU, nn.LeakyReLU],
    use_scales = [True, True, True],
    change_sizes = [0]*3,
    dropout_ps = [0., 0.125, 0.25],
    fc_size = 2,
)

model, optimizer_d, optimizer_g = create_model_and_optimizer(
    model_class = CycleGAN,
    model_params = dict(
        gen_a_b_params = decoder_params,
        gen_b_a_params = decoder_params,
        discr_a_params = encoder_params,
        discr_b_params = encoder_params,
    ),
    lr = 1e-3,
    device = device,
)

def Read_model():
    model_name = "/content/drive/MyDrive/MLBD/ML2/sem5/cycle_gan#0(1)_6"

    checkpoint = torch.load(
        os.path.join("./", f"{model_name}.pt"),
        map_location=device, weights_only=False
    )

    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=100, eta_min=1e-6, verbose=True)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=100, eta_min=1e-6, verbose=True)

    # Загружаем состояния из чекпоинта
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    epoch = checkpoint['epoch']
    plots = checkpoint['plots']

    return model, optimizer_d, optimizer_g





st.title('Uber pickups in NYC')

col1, col2 = st.columns(2)
# If user attempts to upload a file.
#img = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

path = "/content/vangogh2photo/testB/2014-08-01 17:41:55.jpg"
img = Image.open(path)
hyperparams = dict(
    resize=256,
    kernel_size=(5, 7),
    sigma=(0.1, 1.0),
    brightness = 0.2,
    contrast = 0.3,
    saturation = 0.3,
    feature_range=(0., 1.)
)
train_transform_a, val_transform_a, de_normalize_a = get_transforms([0.53160207, 0.55724978, 0.51964466], [0.20851268, 0.20972922, 0.27664949], **hyperparams)
train_transform_b, val_transform_b, de_normalize_b = get_transforms([0.45088902, 0.43960349, 0.40698783], [0.21328361, 0.20770899, 0.21588526], **hyperparams)
img_t = val_transform_a(img)
img_t = img_t.unsqueeze(0)

model, optimizer_d, optimizer_g = Read_model()

import matplotlib.pyplot as plt

if img is not None:
    #file_details = {"Filename":img.name,"FileType":img.type,"FileSize":img.size}
    #st.write(file_details)
    
    with col1:
        st.text("Original Image")
        #print(img.shape)
        st.image(img, width=256)
    with col2:
        st.text("Transformed Image")
        #print(img_t)
        _img = Inference_b(model, img_t, de_normalize_b)#.transpose((2, 0, 1))
        #print(_img.shape)
        #print(_img.shape)
        #plt.imshow(_img)
        #__img = _img - _img.min()
        #___img = Image.fromarray((__img)/__img.max() * 255).astype(np.uint8)
        #blurred_img = apply_convolution(img, kernel_blur, height, width)
        cv2.imwrite('temporary.jpg', _img)
        st.image('temporary.jpg', width=256)
        #st.image(_img, width=256)

    #image = Image.open(img)
    
    #st.image(img,use_container_width=False)