import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class DDUNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, args, num_input_channels=2, num_output_channels=2,
                feature_scale=4, more_layers=0,
                upsample_mode='bilinear', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=False, need_bias=True):
        super(DDUNet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers


        if args.dilation_type == 'constant':
            print('Constant Dilation')
            self.dilation = [[4,4],[4,4],[4,4],[4,4],[4,4],4]
        else:
            print('Exponential Dilation')
            self.dilation = [[1,2],[4,8],[16,32],[32,16],[8,4],2]

        self.transition_dilation = 1
        #filters = [64, 128, 256, 512, 1024]
        filters = [35, 70, 140]
        #filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[0], norm_layer, need_bias, pad, self.dilation[0])
        self.start_transition = conv(num_input_channels+2*filters[0], filters[0], 1, bias=need_bias, pad=pad, dilation=self.transition_dilation)

        self.down1 = unetDown(filters[0], filters[1], norm_layer, need_bias, pad, self.dilation[1])
        self.down1_transition = conv(filters[0]+2*filters[1], filters[1], 1, bias=need_bias, pad=pad, dilation=self.transition_dilation)

        self.down2 = unetDown(filters[1], filters[2], norm_layer, need_bias, pad, self.dilation[2])
        self.down2_transition = conv(filters[1]+2*filters[2], filters[2], 1, bias=need_bias, pad=pad, dilation=self.transition_dilation)
        #self.down3 = unetDown(filters[2], filters[3], norm_layer, need_bias, pad)
        #self.down4 = unetDown(filters[3], filters[4], norm_layer, need_bias, pad)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[2], filters[2], norm_layer, need_bias, pad, self.dilation[2]) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[2], upsample_mode, need_bias, pad, self.dilation[2], same_num_filt =True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        #self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
        #self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp(filters[1], filters[0], upsample_mode, need_bias, pad, self.dilation[3])
        self.up2_transition = conv(filters[0]+filters[1]+2*filters[2], filters[1], 1, bias=need_bias, pad=pad, dilation=self.transition_dilation)

        self.up1 = unetUp(filters[0], num_input_channels, upsample_mode, need_bias, pad, self.dilation[4])
        self.up1_transition = conv(num_input_channels+filters[0]+2*filters[1], filters[0], 1, bias=need_bias, pad=pad, dilation=self.transition_dilation)
        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad, dilation=self.dilation[5])

        if need_sigmoid:
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs):

        # Downsample
        downs = [inputs]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(inputs)
        # if self.counter %2 == 0:
        #     print(torch.max(self.final.weight.data - self.prev))
        # if self.counter == 10:
        #     assert False
        in64_transition = self.start_transition(in64)
        down1 = self.down1(in64_transition)
        down1_transition = self.down1_transition(down1)
        down2 = self.down2(down1_transition)
        down2_transition = self.down2_transition(down2)

        #down3 = self.down3(down2)
        #down4 = self.down4(down3)

        if self.more_layers > 0:
            prevs = [down2]
            for kk, d in enumerate(self.more_downs):
                # print(prevs[-1].size())
                out = d(prevs[-1])
                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_= l(up_, prevs[self.more - idx - 2])
        else:
            up_= down2_transition

        #up4= self.up4(up_, down3)
        #up3= self.up3(up4, down2)
        up2= self.up2(up_, down1)
        temp = self.up2_transition(up2)
        up1= self.up1(temp, in64)
        up1 = self.up1_transition(up1)
        #up1= self.up1(up2, in64)

        return self.final(up1)



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, dilation):
        super(unetConv2, self).__init__()

        if norm_layer is not None:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad, dilation=dilation[0]),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(in_size+out_size, out_size, 3, bias=need_bias, pad=pad, dilation=dilation[1]),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad, dilation=dilation[0]),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(in_size+out_size, out_size, 3, bias=need_bias, pad=pad, dilation=dilation[1]),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs= self.conv1(inputs)
        outputs = torch.cat([outputs, inputs], 1)
        out= self.conv2(outputs)
        outputs = torch.cat([out, outputs], 1)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad, dilation):
        super(unetDown, self).__init__()
        self.conv= unetConv2(in_size, out_size, norm_layer, need_bias, pad, dilation)
        self.down= nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs= self.down(inputs)
        outputs= self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, init_filters, upsample_mode, need_bias, pad, dilation=[2], same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1, dilation=1)
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad, dilation=dilation)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   conv(num_filt, out_size, 3, bias=need_bias, pad=pad, dilation=1))
            self.conv= unetConv2(3*out_size + init_filters, out_size, None, need_bias, pad, dilation)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)


        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output= self.conv(torch.cat([in1_up, inputs2_], 1))
        #output= torch.cat([in1_up, inputs2_], 1)
        return output
