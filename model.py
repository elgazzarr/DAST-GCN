import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class model_static(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.1,gated=False, supports=None, gcn_bool=True, addaptadj=True, residual=True, pool=False, in_dim=2,residual_channels=32,dilation_channels=32,end_channels=32,kernel_size=2,blocks=1,layers=1):
        super(gwnet_static, self).__init__()

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.pool = pool
        self.res = residual
        self.addaptadj = addaptadj
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        self.r = residual_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_len = 0
        self.gated = gated
        self.adjs_s = nn.ParameterList()
        self.adjs_t = nn.ParameterList()
        receptive_field = 1


        self.supports = []
        self.adjs_s = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.adjs_t = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)



        self.supports_len +=1


        #print(self.supports.shape)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions

                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size),dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))


                self.bn.append(nn.BatchNorm2d(residual_channels))

                #self.pool_layers.append(nn.MaxPool2d(1,2))

                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels, dropout=0.1, support_len=self.supports_len))


        #print('recieptive_field = {}'.format(receptive_field))
        self.end_conv_1 = nn.Conv2d(in_channels=residual_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(num_nodes*end_channels,2)




    def forward(self, input):
        # input shape B,D,N,T

        # pad input if receptive field is larger than input
        in_len = input.size(3)
        x = input
        x = self.start_conv(x)
        skip = 0
        new_supports = None


        for i in range(self.blocks * self.layers):

            residual = x  # B,32,N,T

            # Two dilated causal convolution for temporal feature extraction
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            adp = F.softmax(F.relu(torch.mm(self.adjs_s, self.adjs_s.transpose(0,1))), dim=1)
            new_supports = [adp]

            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)


        x = F.relu(self.end_conv_1(x))# B,1,N,T
        x = torch.mean(x,dim=-1)  # B,1,N,1

        x = torch.flatten(x,start_dim=1)
        x =F.dropout(x, self.dropout, training=self.training)

        x = self.last_linear(x)
        return x



class model_dynamic(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.1,gated=False, supports=None, gcn_bool=True, addaptadj=True, residual=True, pool=False, in_dim=2,residual_channels=32,dilation_channels=32,end_channels=32,kernel_size=2,blocks=1,layers=1):
        super(gwnet_dynamic, self).__init__()

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.pool = pool
        self.res = residual
        self.addaptadj = addaptadj
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        self.r = residual_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports_len = 0
        self.gated = gated
        self.adjs_s = nn.ParameterList()
        self.adjs_t = nn.ParameterList()
        receptive_field = 1
        self.supports = None

        for _ in range(layers*blocks):
            nodevec_s = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
            nodevec_t = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
            self.adjs_s.append(nodevec_s)
            self.adjs_t.append(nodevec_t)


        self.supports_len +=1


        #print(self.supports.shape)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions

                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size),dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))


                self.bn.append(nn.BatchNorm2d(residual_channels))

                #self.pool_layers.append(nn.MaxPool2d(1,2))

                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels, dropout=0.1, support_len=self.supports_len))


        #print('recieptive_field = {}'.format(receptive_field))
        self.end_conv_1 = nn.Conv2d(in_channels=residual_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(num_nodes*end_channels,2)



    def forward(self, input):
        # input shape B,D,N,T

        in_len = input.size(3)
        x = input
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):


            residual = x  # B,C,N,T

            # Two dilated causal convolution for temporal feature extraction
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            adp = F.softmax(F.relu(torch.mm(self.adjs_s[i], self.adjs_t[i])), dim=1)
            new_supports = [adp]
            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)


        x = F.relu(self.end_conv_1(x))# B,1,N,T
        x = torch.mean(x,dim=-1)  # B,1,N,1
        x = torch.flatten(x,start_dim=1)
        x =F.dropout(x, self.dropout, training=self.training)
        x = self.last_linear(x)
        return x
