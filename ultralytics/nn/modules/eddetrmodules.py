import math
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3, C2f, AIFI, RepC3

class HTUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, high_pass_type='laplacian'):
        super(HTUnit, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_sizeT
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.high_pass_type = high_pass_type
        
        # 初始化卷积权重
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # 应用高通滤波器初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 根据选择的高通滤波器类型初始化权重
        if self.high_pass_type == 'laplacian':
            # 拉普拉斯高通滤波器初始化
            self._init_laplacian_filter()
        else:
            raise ValueError(f"Unsupported high pass filter type: {self.high_pass_type}")
            
        # 初始化偏置
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _init_laplacian_filter(self):
        """初始化类似拉普拉斯算子的高通滤波器"""
        # 创建拉普拉斯核心
        center = self.kernel_size // 2
        for i in range(self.out_channels):
            for j in range(self.in_channels // self.groups):
                kernel = torch.zeros(self.kernel_size, self.kernel_size)
                kernel[center, center] = 1.0
                for k in range(self.kernel_size):
                    for l in range(self.kernel_size):
                        if k != center or l != center:
                            kernel[k, l] = -1.0 / ((self.kernel_size * self.kernel_size) - 1)
                self.weight.data[i, j] = kernel

    def forward(self, x):
        # 应用高通滤波器卷积
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
        return y
        
class LSUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True):
        super(LSUnit, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def func(self , x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

        weights_c = weights.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc


class DFEA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True,):
        super(DAConv, self).__init__()
        self.padding = padding
        self.stride = strideDFEA
        self.bias = bias
        self.inp_dim=in_channels
        self.out_dim=out_channels
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 

        self.cac = HTUnit(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
                              ,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.cdc = LSUnit(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
                              , stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.register_parameter('cac_theta', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('cdc_theta', nn.Parameter(torch.tensor(1.0)))
        # print("Using {} \n  ".format(self.__class__.__name__))

    def forward(self,x):

        return torch.sigmoid(self.cac_theta)*self.cac(x) \
                   + torch.sigmoid(self.cdc_theta)*self.cdc(x)


    def re_para(self):
        k = self.cac.weight.data.sum( dim=[2, 3] )
        loc = int(self.cac.weight.size(3) /2)
        cac_k = torch.clone (self.cac.weight.data)
        cac_k[:,:,loc,loc] += k

        k = self.cdc.weight.data.sum(dim=[2, 3])
        loc = int(self.cdc.weight.size(3) / 2)
        cdc_k = torch.clone(self.cdc.weight.data)
        cdc_k[:, :, loc, loc] -= k

        self.K = torch.sigmoid(self.cac_theta)*cac_k + torch.sigmoid(self.cdc_theta)*cdc_k
        self.K = self.K.to(self.cac.weight.device)

        if self.cac.bias  is not None:
            self.B = torch.sigmoid(self.cac_theta)*self.cac.bias.data \
                     + torch.sigmoid(self.cdc_theta)*self.cdc.bias.data
            self.B = self.B.to(self.cac.weight.device)
        else:
            self.B = None

    def test_forward(self,x):
        self.re_para()
        self.K = self.K.to(x.device)
        if self.B is not None:
            self.B = self.B.to(x.device)
        return F.conv2d(input=x , weight=self.K,bias=self.B,padding=self.padding,stride=self.stride)


class EDFEA(nn.Module):
    """双分支局部特征增强"""
    def __init__(self, in_dim):
        super().__init__()
        self.da = DFEA(in_dim, in_dim, 3, 1, 1)
        self.aug = EG(in_dim)
    
    def forward(self, x):
        return self.aug(self.da(x))

class EG(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim,act=nn.Sigmoid())
        self.pool = nn.MaxPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x-self.out_conv(edge)
        return x + edge

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        
        self.fee1 = EDFEA(c1)
        self.fee2 = EDFEA(c1)
        self.conv = nn.Conv2d(c1, c2, 1, 1)
        self.add = True if c1==c2 else False

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.conv(self.fee2(self.fee1(x))) if self.add else self.conv(self.fee2(self.fee1(x)))


class EDFEABlock(C2f):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        # self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, e=1) for _ in range(n)))
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, e=1) for _ in range(n))
