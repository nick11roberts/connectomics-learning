import torch
import torch.nn as nn
from torch.autograd import Variable

class PadChannels(nn.Module):
    def __init__(self, in_planes, planes):
        super(PadChannels, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.pad = self.planes - self.in_planes
        self.padding = None

    def __repr__(self):
        return (self._get_name() + '('
                + str(self.in_planes) + ', '
                + str(self.planes) + ')')

    def forward(self, x):
        if x.is_cuda:
            self.padding = Variable(torch.zeros(x.size()[0],
                                                self.pad,
                                                x.size()[2],
                                                x.size()[3]).cuda())
        else:
            self.padding = Variable(torch.zeros(x.size()[0],
                                                self.pad,
                                                x.size()[2],
                                                x.size()[3]))

        self.padding.requires_grad = False

        x = torch.cat((x, self.padding), 1)
        return x
