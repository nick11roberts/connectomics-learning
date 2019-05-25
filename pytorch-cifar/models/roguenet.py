'''RogueNet in PyTorch.

UnifyID AI Fellowship 2019 project
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
import models.pad as pad
from scipy.sparse import diags
from scipy.linalg import circulant

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class WattsStrogatzModel():
    def __init__(self):
        pass

    def _distance_matrix(self, L):
        Dmax = L // 2

        D  = list(range(Dmax+1))
        D += D[-2 + (L % 2):0:-1]

        return circulant(D) / Dmax

    def _pd(self, d, p0, beta):
        return beta * p0 + (d <= p0) * (1 - beta)

    def generate(self, L, p0, beta, directed=False, rngseed=1):
        """
        Watts-Strogatz model of a small-world network

        This generates the full adjacency matrix, which is not a good way to
        store things if the network is sparse.

        Parameters
        ----------
        L        : int
                   Number of nodes.

        p0       : float
                   Edge density. If K is the average degree then p0 = K/(L-1).
                   For directed networks "degree" means out- or in-degree.

        beta     : float
                   "Rewiring probability."

        directed : bool
                   Whether the network is directed or undirected.

        rngseed  : int
                   Seed for the random number generator.

        Returns
        -------
        A        : (L, L) array
                   Adjacency matrix of a WS (potentially) small-world network.

        """
        rng = np.random.RandomState(rngseed)

        d = self._distance_matrix(L)
        p = self._pd(d, p0, beta)

        if directed:
            A = 1 * (rng.random_sample(p.shape) < p)
            np.fill_diagonal(A, 0)
        else:
            upper = np.triu_indices(L, 1)

            A          = np.zeros_like(p, dtype=int)
            A[upper]   = 1 * (rng.rand(len(upper[0])) < p[upper])
            A.T[upper] = A[upper]

        #print("L:   ", L)
        #print("p0:  ", p0)
        #print("beta:", beta)
        print(A)
        return A


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes_all, planes, stride=1, masks=[], layernum=0,
                 stride_table=np.array([]), skip_connections_table={}):
        super(BasicBlock, self).__init__()
        self.layernum = layernum
        self.masks = masks
        in_planes = in_planes_all[-1]

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # TODO add batch normalization here?
        self.conv_gate = nn.Conv2d(planes, planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.conv_gate.weight.requires_grad = True

        self.neuron_in = nn.Parameter(
            torch.ones(len(self.masks)) / np.sqrt(len(self.masks)))

        self.shortcuts = []
        for i in range(layernum + 1):
            shortcut = nn.Sequential()
            if (stride != 1 or in_planes_all[i] != (self.expansion * planes)):
                # TODO making this parameterless might be better
                # i.e. zero filling new channels
                # although, we would still need to deal with stride differences
                shortcut = nn.Sequential(
                    nn.MaxPool2d(kernel_size=1,
                                 stride=int(stride_table[i, layernum])),
                    pad.PadChannels(in_planes_all[i], self.expansion * planes)
                )

            # Modify the skip connections table from the RogueNet module
            skip_connections_table["({},{})".format(i, layernum)] = shortcut
            self.shortcuts.append(shortcut)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #shortcut = self.shortcuts[-1]
        #out += shortcut(x)
        out = F.relu(out)
        return out

    def forward_all(self, xs):
        x = xs[-1]
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        out_conn = [(self.shortcuts[-1](x))]
        for i in range(len(xs) - 1):
            if self.masks[i]:
                shortcut = self.shortcuts[i]
                out_conn.append(self.neuron_in[i] * shortcut(xs[i]))

        #out = (((out + x) * torch.sigmoid(self.conv_gate(out_conn)))
        #       + (out_conn * (1 - torch.sigmoid(self.conv_gate(out_conn)))))

        out += sum(out_conn)
        #out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes_all, planes, stride=1, masks=[], layernum=0,
                 stride_table=np.array([]), skip_connections_table={}):
        super(Bottleneck, self).__init__()
        self.layernum = layernum
        self.masks = masks
        in_planes = in_planes_all[-1]

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1,
                               bias=False)


        self.conv_gate = nn.Conv2d(self.expansion*planes,
                                   self.expansion*planes,
                                   kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.shortcuts = []
        for i in range(layernum + 1):
            shortcut = nn.Sequential()
            if (stride != 1 or in_planes_all[i] != (self.expansion * planes)):
                # TODO making this parameterless might be better
                # i.e. zero filling new channels
                # although, we would still need to deal with stride differences
                shortcut = nn.Sequential(
                    nn.MaxPool2d(kernel_size=1,
                                 stride=int(stride_table[i, layernum])),
                    pad.PadChannels(in_planes_all[i], self.expansion * planes)
                )

            # Modify the skip connections table from the RogueNet module
            skip_connections_table["({},{})".format(i, layernum)] = shortcut
            self.shortcuts.append(shortcut)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcuts[-1]
        #out += shortcut(x)
        out = F.relu(out)
        return out

    def forward_all(self, xs):
        x = xs[-1]
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(self.bn3(out))

        out_conn = [(self.shortcuts[-1](x))]
        for i in range(len(xs) - 1):
            if self.masks[i]:
                shortcut = self.shortcuts[i]
                out_conn.append(self.neuron_in[i] * shortcut(xs[i]))

        #out = (((out + x) * torch.sigmoid(self.conv_gate(out_conn)))
        #       + (out_conn * (1 - torch.sigmoid(self.conv_gate(out_conn)))))

        out += sum(out_conn)
        #out = F.relu(out)
        return out


class RogueNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3,
                 K=6, beta=0.06, seed=42):
        super(RogueNet, self).__init__()

        self.full_skip = True
        self.resnet_mode_debug = False
        self.smallworld = WattsStrogatzModel()
        self.K = K
        self.beta = beta
        self.seed = seed
        self.N = sum(num_blocks) + 1

        # Check for valid Watts Strogatz parameters
        assert(self.N > self.K)
        assert(self.K > np.log(self.N))
        assert(np.log(self.N) > 1.0)

        assert((0 <= self.beta) and (self.beta <= 1))

        # Initial Watts Strogatz permutation of the skip connnections
        self.permute()

        self.in_planes = 64
        self.in_planes_all = []
        self.strides = [1,2,2,2]
        self.stride_table = self._gen_stride_table(num_blocks, self.strides)
        self.skip_connections_table = {}

        assert len(num_blocks) == len(self.strides)

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)

        layer1 = self._make_layer(block, 64, num_blocks[0],
                                  self.skip_connections_table,
                                  stride=self.strides[0],
                                  layernum=sum(num_blocks[:0]))
        layer2 = self._make_layer(block, 128, num_blocks[1],
                                  self.skip_connections_table,
                                  stride=self.strides[1],
                                  layernum=sum(num_blocks[:1]))
        layer3 = self._make_layer(block, 256, num_blocks[2],
                                  self.skip_connections_table,
                                  stride=self.strides[2],
                                  layernum=sum(num_blocks[:2]))
        layer4 = self._make_layer(block, 512, num_blocks[3],
                                  self.skip_connections_table,
                                  stride=self.strides[3],
                                  layernum=sum(num_blocks[:3]))

        self.all_blocks = nn.ModuleList(layer1 + layer2 + layer3 + layer4)

        self.all_connections = nn.ModuleDict(self.skip_connections_table)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def permute(self):
        self.skips = self.smallworld.generate(L=self.N,
                                              p0=(self.K / (self.N - 1)),
                                              beta=self.beta,
                                              directed=False,
                                              rngseed=self.seed)
        self.seed += 1

        # TODO possibly remove this, it is for debugging
        # we wts that this is similar to ResNet before randomly
        # permuting connections
        ringlat = diags([0, 1, 1], [0, 1, self.N - 1],
                        shape=(self.N, self.N)).toarray()
        ringlat += ringlat.T

        if not self.full_skip:
            self.skips[0, self.N - 1] = 0.0
            self.skips[1, self.N - 1] = 0.0
            self.skips[0, self.N - 2] = 0.0
            self.skips[self.N - 1, 0] = 0.0
            self.skips[self.N - 1, 1] = 0.0
            self.skips[self.N - 2, 0] = 0.0
            #print(self.skips)

        if self.resnet_mode_debug:
            self.skips = ringlat

        # TODO This is temporary
        #self.skips = np.ones_like(self.skips)
        #self.skips = np.zeros_like(self.skips)

    def _gen_stride_table(self, num_blocks, strides):
        n = np.sum(num_blocks)
        st = np.ones([n, n])
        strides_init = np.ones(n)

        # Populate initial values (first col, first row)
        for nb, i in zip(num_blocks, np.arange(len(num_blocks))):
            strides_init[sum(num_blocks[:i])] = strides[i]
            st[i,i]

        # Compute strides between various layers
        for i in range(n):
            st[i,i] = strides_init[i]
            for j in range(i):
                st[i,j] = np.prod(strides_init[j:i+1])
                st[j,i] = st[i,j]

        return st

    def _make_layer(self, block, planes, num_blocks,
                    skip_connections_table, stride, layernum):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(num_blocks):
            modulenum = layernum + i
            self.in_planes_all.append(self.in_planes)
            layers.append(block(self.in_planes_all,
                                planes, strides[i],
                                self.skips[modulenum+1, :modulenum+1],
                                modulenum,
                                self.stride_table,
                                skip_connections_table))
            self.in_planes = planes * block.expansion
        return layers
        #return layers # nn.Sequential(*layers) # Possibly change this

    def forward(self, x):
        activations = []

        out = self.conv1(x)

        '''
        for i in range(len(self.all_blocks)):
            out = self.all_blocks[i](out)
        '''

        activations.append(out)
        for block_i in self.all_blocks:
            out = block_i.forward_all(activations)
            activations.append(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RogueNet18():
    return RogueNet(BasicBlock, [2,2,2,2])

def RogueNet34():
    return RogueNet(BasicBlock, [3,4,6,3])

def RogueNet50():
    return RogueNet(Bottleneck, [3,4,6,3])

def RogueNet101():
    return RogueNet(Bottleneck, [3,4,23,3])

def RogueNet152():
    return RogueNet(Bottleneck, [3,8,36,3])

def test():
    net = RogueNet152()
    print(net)
    print(count_parameters(net))
    x = torch.randn(1,3,32,32)
    y = net(x)

    # Save onnx model for visualization
    torch.onnx.export(net, x, "onnx/roguenet152.onnx")
    print('Model saved to disk')

#test()
