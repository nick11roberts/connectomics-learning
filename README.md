# Connectomics Learning 

## pytorch-cifar
The files `main.py` and `models/roguenet.py` are the ones of notable interest related to our experiments. `models/roguenet.py` contains the architecture described in our paper, and also contains a utility class for generating Watts Strogatz random graphs for use within the RogueNet architecture. These files are based on code from https://github.com/kuangliu/pytorch-cifar.

## RandWireNN
These files correspond to the experiments related to the RamanujanNet architecture described in the paper. This is based on code from https://github.com/seungwonpark/RandWireNN, and is run in the same manner (much of the documentation can be inherited from that repository). We have included Ramanujan graphs with various spectral properties within `model/graphs/generated`.

## CElegansNN
These files correspond to the experiments related to the CElegansNN architecture described in the paper, again based on code from https://github.com/seungwonpark/RandWireNN, and thus inherits much of the associated documentation. Here, `model/graphs/generated` includes the graph of the C. Elegans neuronal network to be loaded into the model. Notably, `model/model.py` includes modifications from the original codebase from user `seungwonpark` for freezing the convolutional and fully connected layers such that the only learnable parameters are the edge weights of the C. Elegans neuronal network, as well as including only one instance of a DAGLayer.
