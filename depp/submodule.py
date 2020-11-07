import torch
import torch.distributions.normal as normal
from torch import nn
from typing import List, Callable

class stack_conv(nn.Module):
    def __init__(self,
                 input_channel: int,
                 channels: List[int],
                 strides: List[int],
                 kernel_sizes: List[int],
                 activation: Callable = nn.CELU,
                 paddings=None):
        super(stack_conv, self).__init__()

        layers = []

        if not paddings:
            paddings = [0] * len(input_channel)

        for i, (c, k, s, p) in enumerate(zip(channels, kernel_sizes, strides, paddings)):
            if i == 0:
                layers.append(nn.Conv1d(input_channel, c, k, s, p))
            else:
                layers.append(nn.Conv1d(channels[i-1], c, k, s, p))
            if i < len(channels) - 1:
                layers.append(activation())
                layers.append(nn.BatchNorm1d(c))

        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)

        return x

class stack_convtran(nn.Module):
    def __init__(self,
                 input_channel: int,
                 channels: List[int],
                 strides: List[int],
                 kernel_sizes: List[int],
                 activation: Callable = nn.CELU):
        super(stack_convtran, self).__init__()

        layers = []

        for i, (c, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            if i == 0:
                layers.append(nn.ConvTranspose1d(input_channel, c, k, s))
            else:
                layers.append(nn.ConvTranspose1d(channels[i-1], c, k, s))
            if i < len(channels) - 1:
                layers.append(activation())
                layers.append(nn.BatchNorm1d(c))

        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)

        return x


class resblock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, ratio, groups=1):
        super(resblock, self).__init__()
        self.layers = nn.Sequential(
            # nn.GroupNorm(num_groups=groups, num_channels=in_channel),
            nn.CELU(),
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=(kernel_size-1)//2, groups=groups),
            # nn.GroupNorm(num_groups=groups, num_channels=out_channel),
            nn.CELU(),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=(kernel_size-1)//2, groups=groups)
        )

        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_out = self.layers(x)

        return self.ratio * x_out + x

# following module is modified from https://github.com/pclucas14/iaf-vae/blob/master/layers.py

class MaskedConv(nn.Conv1d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, L = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, L // 2 + (mask_type == 'B'):] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super(MaskedConv, self).forward(x)

class ARMultiConv(nn.Module):
    def __init__(self, n_h, n_out, args, activation=nn.CELU):
        super(ARMultiConv, self).__init__()
        self.activation = activation()

        convs, out_convs = [], []

        for i in range(n_h):
            if i == 0:
                convs.append(MaskedConv('A', 1, args.h_channel, 3, 1, 1))
            else:
                convs.append(MaskedConv('B', args.h_channel, args.h_channel, 3, 1, 1))

        for _ in range(n_out):
            out_convs.append(MaskedConv('B', args.h_channel, 1, 3, 1, 1))

        self.convs = nn.ModuleList(convs)
        self.out_convs = nn.ModuleList(out_convs)

    def forward(self, x, context):
        # x (bs, 1, args.embedding_size)
        # context (bs, 1, args.embedding_size)
        for i, conv_layer in enumerate(self.convs):
            x = conv_layer(x)
            if i == 0:
                x += context
            x = self.activation(x)

        return [conv_layer(x) for conv_layer in self.out_convs]

class IAFLayer(nn.Module):
    def __init__(self, args):
        super(IAFLayer, self).__init__()
        n_in = 1
        n_out = 4

        self.embedding_size = args.embedding_size
        self.h_channel = args.h_channel
        self.iaf = args.iaf
        self.args = args

        self.down_conv_b = nn.Sequential(
            nn.Conv1d(self.h_channel + 1, args.h_channel, 3, 1, 1),
            nn.BatchNorm1d(args.h_channel)
        )

        self.down_conv_a = nn.Sequential(
            nn.Conv1d(args.h_channel, 1 * 2 + 1 * 2 + 1 + args.h_channel, 3, 1, 1),
            # nn.BatchNorm1d(1 * 2 + 1 * 2 + 1 + args.h_channel)
        )

        self.up_conv_a = nn.Sequential(
            nn.Conv1d(args.h_channel, 1 * 2 + 1 + args.h_channel, 3, 1, 1),
            # nn.BatchNorm1d(1 * 2 + 1 + args.h_channel)
        )

        self.up_conv_b = nn.Sequential(
            nn.Conv1d(args.h_channel, args.h_channel, 3, 1, 1),
            nn.BatchNorm1d(args.h_channel)
        )

        if args.iaf:
            self.down_ar_conv = ARMultiConv(2, 2, args)

        self.celu = nn.CELU()

        self.args = args

    def up(self, input):
        x = self.up_conv_a(input)

        self.qz_mean, self.qz_std, self.up_context, h = x.split([1, 1, 1, self.args.h_channel], 1)
        self.qz_std = torch.nn.functional.softplus(self.qz_std)

        h = self.celu(h)
        h = self.up_conv_b(h)

        return input + 0.1 * h

    def down(self, input, sample=False):
        bs = input.shape[0]
        device = input.device
        x = self.celu(input)
        x = self.down_conv_a(x)

        pz_mean, pz_std, rz_mean, rz_std, down_context, h_det = x.split([1] * 5 + [self.args.h_channel], 1)
        pz_std = torch.nn.functional.softplus(pz_std)
        rz_std = torch.nn.functional.softplus(rz_std)
        prior = normal.Normal(loc=pz_mean, scale=pz_std+1e-4)

        if sample:
            z = prior.rsample()
            kl = torch.zeros(bs).to(device)
        else:
            posterior = normal.Normal(loc=rz_mean+self.qz_mean,
                                      scale=rz_std + self.qz_std+1e-4)

            hard_encode = rz_mean + self.qz_mean

            z = posterior.rsample()
            logqs = posterior.log_prob(z)
            context = self.up_context + down_context

            if self.iaf:
                x = self.down_ar_conv(z, context)
                arw_mean, arw_std = x[0] * 0.1, x[1] * 0.1
                arw_std = torch.nn.functional.softplus(arw_std)
                z = (z - arw_mean) / (arw_std + 1e-4)
                # z = arw_mean + z * torch.exp(arw_logsd)
                # z = (z - arw_mean) / 0.001
                logqs += torch.log(arw_std + 1e-4)

                x_hard = self.down_ar_conv(rz_mean+self.qz_mean, context)
                arw_mean_hard, arw_std_hard = x_hard[0] * 0.1, x_hard[1] * 0.1
                arw_std_hard = torch.nn.functional.softplus(arw_std_hard)

                hard_encode = (rz_mean + self.qz_mean - arw_mean_hard) / (arw_std_hard + 1e-4)
                # hard_encode = (rz_mean + self.qz_mean) * torch.exp(arw_logsd_hard) + arw_mean_hard

            logps = prior.log_prob(z)
            # logps = torch.ones_like(logqs)
            kl = logqs - logps

        h = torch.cat((z, h_det), 1)
        h = self.celu(h)

        h = self.down_conv_b(h)

        return input + 0.1 * h, kl, z, hard_encode