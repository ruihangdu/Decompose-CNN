from torch import nn

def decomp_alexnet(net, rank_func, decomp_func):
    i = 1
    
    while i < len(net.features):
        # find out the rank of the first conv layer
        layer_i = net.features[i]

        if not isinstance(layer_i, nn.Conv2d):
            i += 1
            continue
        
        layer_i = net.features[i]
        rank = rank_func(layer_i)
        print('rank of the {}th layer: {}'.format(i, rank))
        
        # debugging
        print("begin decomposing layer {}".format(i))
        decomp_layers = decomp_func(layer_i, rank)
        print("finished decomposing layer {}".format(i))

        net.features = nn.Sequential(\
        *(list(net.features[:i]) + decomp_layers + list(net.features[i + 1:])))

        i +=  len(decomp_layers)

    return net
