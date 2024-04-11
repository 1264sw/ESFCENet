# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.ops import DeformConv2D
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant, XavierUniform

__all__ = ["ResNet_vd", "ConvBNLayer", "DeformableConvV2"]


class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 skip_quant=False,
                 dcn_bias_regularizer=L2Decay(0.),
                 dcn_bias_lr_scale=2.):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size**2 * groups
        self.mask_channel = kernel_size**2 * groups

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = ParamAttr(
                initializer=Constant(value=0),
                regularizer=dcn_bias_regularizer,
                learning_rate=dcn_bias_lr_scale)
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            deformable_groups=groups,
            weight_attr=weight_attr,
            bias_attr=dcn_bias_attr)

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = ParamAttr(initializer=Constant(0.))
        else:
            offset_bias_attr = ParamAttr(
                initializer=Constant(0.),
                learning_rate=lr_scale,
                regularizer=regularizer)
        self.conv_offset = nn.Conv2D(
            in_channels,
            groups * 3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=offset_bias_attr)
        if skip_quant:
            self.conv_offset.skip_quant = True

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 dcn_groups=1,
                 is_vd_mode=False,
                 act=None,
                 is_dcn=False):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        if not is_dcn:
            self._conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias_attr=False)
        else:
            self._conv = DeformableConvV2(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=dcn_groups,  #groups,
                bias_attr=False)
        self._batch_norm = nn.BatchNorm(out_channels, act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            shortcut=True,
            if_first=False,
            is_dcn=False, ):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu')
        # self.conv1 = ConvBNLayer(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     stride=stride,
        #     act='relu',
        #     is_dcn=is_dcn,
        #     dcn_groups=2)
        self.conv1=ACmix(out_channels,out_channels,stride=stride)
        self.batch_norm = nn.BatchNorm(out_channels, act='relu')
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        # 这里加了ACmix
        conv1 = self.conv1(y)
        #后面需要跟BN+relu
        conv1 = self.batch_norm(conv1)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            shortcut=True,
            if_first=False, ):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu')
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y


class ResNet_vd(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 layers=50,
                 dcn_stage=None,
                 out_indices=None,
                 **kwargs):
        super(ResNet_vd, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.dcn_stage = dcn_stage if dcn_stage is not None else [
            False, False, False, False
        ]
        self.out_indices = out_indices if out_indices is not None else [
            0, 1, 2, 3
        ]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu')
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu')
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.stages = []
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                is_dcn = self.dcn_stage[block]
                for i in range(depth[block]):
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            is_dcn=is_dcn))
                    shortcut = True
                    block_list.append(bottleneck_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0))
                    shortcut = True
                    block_list.append(basic_block)
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block])
                self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            if i in self.out_indices:
                out.append(y)
        return out


def position(H, W):
    loc_w = paddle.repeat_interleave(paddle.linspace(-1.0, 1.0, W).unsqueeze(0), repeats=H, axis=0)
    loc_h = paddle.repeat_interleave(paddle.linspace(-1.0, 1.0, H).unsqueeze(1), repeats=W, axis=1)
    loc = paddle.concat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], axis=0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


class ACmix(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = self.create_parameter([1], default_initializer=nn.initializer.Assign([0.5]))
        self.rate2 = self.create_parameter([1], default_initializer=nn.initializer.Assign([0.5]))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2D(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2D(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2D(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2D(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.Pad2D(self.padding_att, mode='reflect')
        self.unfold = nn.Unfold(kernel_sizes=kernel_att, strides=self.stride, paddings=0)
        self.softmax = nn.Softmax(axis=1)

        self.fc = nn.Conv2D(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias_attr=False)
        self.dep_conv = nn.Conv2D(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias_attr=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        kernel = paddle.zeros((self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv))
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = paddle.repeat_interleave(kernel.unsqueeze(0), self.out_planes, axis=0)
        self.dep_conv.weight = self.create_parameter((self.out_planes, 1, 1, 1),
                                                     default_initializer=nn.initializer.Assign(kernel))
        self.dep_conv.bias = self.create_parameter([1], default_initializer=nn.initializer.Assign([0.]))

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # att
        # positional encoding
        pe = self.conv_p(position(h, w))

        q_att = q.reshape((b * self.head, self.head_dim, h, w)) * scaling
        k_att = k.reshape((b * self.head, self.head_dim, h, w))
        v_att = v.reshape((b * self.head, self.head_dim, h, w))

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        # b*head, head_dim, k_att^2, h_out, w_out
        unfold_k = self.unfold(self.pad_att(k_att)).reshape(
            (b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out))
        # 1, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).reshape(
            (1, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out))

        # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).reshape(
            (b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out))
        out_att = (att.unsqueeze(1) * out_att).sum(2).reshape((b, self.out_planes, h_out, w_out))

        # conv
        f_all = self.fc(paddle.concat([q.reshape((b, self.head, self.head_dim, h * w)),
                                       k.reshape((b, self.head, self.head_dim, h * w)),
                                       v.reshape((b, self.head, self.head_dim, h * w))], axis=1))
        f_conv = f_all.transpose([0, 2, 1, 3]).reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))

        out_conv = self.dep_conv(f_conv)
        return self.rate1 * out_att + self.rate2 * out_conv