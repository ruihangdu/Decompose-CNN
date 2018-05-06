## Goal
The goal of this program is to decompose each convolutional layers in a model to reduce the total number of floating-point operations (I'll use the shorthand flops) in the convolutions as well as the number of parameters in the model.

## Contributions
This is an extension of the work https://github.com/jacobgil/pytorch-tensor-decompositions.
In this implementation, everything, including finding the ranks and the actual CP/Tucker Decomposition, is done in PyTorch without switching to numpy.

## CNN architecture decomposed
- [x] AlexNet
- [ ] VGG
- [x] ResNet50

## Dataset used
- ImageNet ILSVRC2012 dataset

## Usage
```bash
python3 scripts/decomp.py [-p PATH] [-d DECOMPTYPE] [-m MODEL] [-r CHECKPOINT] [-s STATEDICT] [-v]
```
- PATH specifies the path to the dataset
- DECOMPTYPE is either cp (default) or tucker
- If a model is already decomposed, it could be passed in as the MODEL parameter (By default, the Torchvision pretrained ResNet50 is used).
- If continue a fine-tuning from a checkpoint, pass in the checkpoint as CHECKPOINT
- To specify the parameters for the model, use STATEDICT
- [-v] option for evaluating the inference accuracy of the model without fine-tuning

## Pre-decomposed and fine-tuned model

A pre-decomposed ResNet50 is included in the models directory as resnet50_tucker.pth.

The fine-tuned parameters for the model is the resnet50_tucker_state.pth in the models directory.

## Results

It turn out that Tucker decomposition yields lower accuracy loss than CP decomposition in my experiments, so the results below are all from Tucker decomposition.

### AlexNet

|  | Top-1 | Top-5 | flops in convolutions (Giga) |
| ------------- | ------------- | ------------- |  ------------- |
| Before | 56.55% | 79.09% | 1.31 |
| After | 54.90% | 77.90% | 0.45 |

### ResNet50

|  | Top-1 | Top-5 | flops in convolutions (Giga) |
| ------------- | ------------- | ------------- |  ------------- |
| Before | 76.15% | 92.87% | 7.0 |
| After | 74.88% | 92.39% | 4.7 |

# References

- CP-decomposition with Tensor Power Method for Convolutional Neural Networks Compression: https://arxiv.org/abs/1701.07148
- Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications: https://arxiv.org/abs/1511.06530
- PyTorch CP and Tucker decomposition: https://github.com/jacobgil/pytorch-tensor-decompositions
- VBMF code: https://github.com/CasvandenBogaard/VBMF
- Tensorly: https://github.com/tensorly/tensorly

### Any comments, thoughts, and improvements are appreciated
