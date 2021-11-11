import torch.optim as optim
from model import *
import util
import torch.nn as nn
from scheduler import CosineAnnealingWarmupRestarts


class trainer():

    def __init__(self, dynamic, in_dim, kernel, num_nodes, filters , dropout, lrate, wdecay, device, supports, blocks):
        if dynamic:
            self.model = model_dynamic(device, num_nodes, dropout, blocks=blocks, supports=supports, gcn_bool=True, addaptadj=True,  in_dim=in_dim, residual_channels=filters, dilation_channels=filters, kernel_size=kernel, end_channels=1)
        else:
            self.model = model_dynamic(device, num_nodes, dropout, blocks=blocks, supports=supports, gcn_bool=True, addaptadj=True,  in_dim=in_dim, residual_channels=filters, dilation_channels=filters, kernel_size=kernel, end_channels=1)

        self.model.to(device)
        self.lrate = lrate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                                  first_cycle_steps=50,
                                                  cycle_mult=1.0,
                                                  max_lr=0.01,
                                                  min_lr=0.001,
                                                  warmup_steps=10,
                                                  gamma=0.5)
        self.loss = nn.CrossEntropyLoss().to(device)
        self.clip = None

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input)
        real = torch.max(real_val,1)[1]
        predict = output

        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return loss.item()

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        real = torch.max(real_val,1)[1]
        predict = output
        loss = self.loss(predict, real)
        return loss.item()

