from functools import reduce
from torch import nn

def decomp_resnet(net, rank_func, decomp_func):
    mulfunc = (lambda x,y:x*y)
    for n, m in net.named_children():
        num_children = sum(1 for i in m.children())
        if num_children != 0:
            # in a layer of resnet
            layer = getattr(net, n)
            # decomp every bottleneck
            for i in range(num_children):
                bottleneck = layer[i]
                conv2 = getattr(bottleneck, 'conv2')

                rank = rank_func(conv2)

                if type(rank) == int:
                    # in this case cp decomp is used
                    reduced = rank**2
                else:
                    # tucker decomp in this case
                    reduced = reduce(mulfunc, rank)

                if reduced < \
                reduce(mulfunc, [conv2.in_channels, conv2.out_channels]):
                    print('ranks for bottleneck {} in {}: {}'\
                    .format(i, n, rank))

                    new_layers = decomp_func(conv2, rank) 

                    setattr(bottleneck, 'conv2', nn.Sequential(*new_layers))

                del conv2
                del bottleneck
            del layer
    return net
