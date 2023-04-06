import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:
                    # print(grad)
                    # TODO
                    tmp = param_t - lr_inner * grad
                    self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)


class FusionNet(MetaModule):
    def __init__(self, block_num, feature_out):
        super(FusionNet, self).__init__()
        block1 = []
        self.feature_out = feature_out
        for i in range(block_num):
            if i == 0:
                block1.append(FusionBlock(in_block=4, out_block=64, k_size=3))
            elif i == 1:
                block1.append(FusionBlock(in_block=128, out_block=128, k_size=3))
            else:
                block1.append(FusionBlock(in_block=256, out_block=128, k_size=3))
        self.block1 = nn.Sequential(*block1)

        if block_num == 1:
            self.block2_in = 128
        else:
            self.block2_in = 256
        self.block2 = nn.Sequential(
            nn.Conv2d(self.block2_in, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 2, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):

        if self.feature_out:
            block_out = []
            for i in range(len(self.block1)):
                x = self.block1[i](x)
                block_out.append(x.clone())

            x = self.block2(x)
            return block_out, x  # [from shallow to deep]
        else:
            x = self.block1(x)
            x = self.block2(x)

            return None, x


class FusionBlock(MetaModule):
    def __init__(self, in_block, out_block, k_size=3):
        super(FusionBlock, self).__init__()
        self.conv1_1 = MetaConv2d(
            in_channels=in_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_2 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_0_00 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.conv1_0_01 = MetaConv2d(
            in_channels=out_block,
            out_channels=out_block,
            kernel_size=k_size,
            stride=1,
            padding=(k_size - 1) // 2,
            bias=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)

        x0 = self.conv1_0_00(x)
        x0 = self.relu(x0)
        x1 = self.conv1_0_01(x)
        x1 = self.relu(x1)

        return torch.cat([x0, x1], dim=1)
