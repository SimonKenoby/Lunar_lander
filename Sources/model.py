import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) 
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4])*weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out)) 
        m.weight.data.uniform_(-w_bound, w_bound)
        m.biais.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out)) 
        m.weight.data.uniform_(-w_bound, w_bound)
        m.biais.data.fill_(0)

class AC_netwok(torch.nn.Module):

    def __init__(self, s_space, a_space):
        super(AC_netwok, self).__init__()
        self.conv1 = nn.Conv2d(s_space, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
