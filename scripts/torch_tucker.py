import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import tucker, partial_tucker

def tucker_decomp(layer, rank):
    W = layer.weight.data

    core, [last, first] = partial_tucker(W, modes=[0,1], ranks=rank, init='svd')

    first_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    core_layer = nn.Conv2d(in_channels=core.shape[1],
                                       out_channels=core.shape[0],
                                       kernel_size=layer.kernel_size,
                                       stride=layer.stride,
                                       padding=layer.padding,
                                       dilation=layer.dilation,
                                       bias=False)

    last_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=True)
    
    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    fk = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    lk = last.unsqueeze_(-1).unsqueeze_(-1)

    first_layer.weight.data = fk
    last_layer.weight.data = lk
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return new_layers


