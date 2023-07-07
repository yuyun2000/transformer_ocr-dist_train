import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DetrConfig, DetrModel
from torch.nn import init
import numpy as np
import math
import torchvision

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备
class model(nn.Module):
    def __init__(self,num):
        super().__init__()

        # self.encoder = DetaModel.from_pretrained( "jozhang97/deta-swin-large-o365",num_queries = 40,two_stage=False,cache_dir='./data/cache')
        # self.encoder = DetrModel.from_pretrained("facebook/detr-resnet-50",cache_dir='./data/cache')
        configuration = DetrConfig(num_queries = 30)
        self.encoder = DetrModel(configuration)
        #self.conv = nn.Conv1d(100,40,1,1,0)
        self.clf = nn.Linear(256, num) #3852 训练集字符数量 +1其他所有未包含的字符 +1开始标志 +1结束标志

        # self.ln2 = nn.BatchNorm1d(30)
        # self.rele = nn.ReLU6()

    def forward(self, inputs,mask):

        # print(mask.shape)
        # mask = mask.squeeze(1)
        out = self.encoder(inputs)
        # out = self.ln2(out['last_hidden_state'])
        # out = self.rele(out)
        #out = self.conv(out)
        # print(out)

        # print(out[0])
        # out1 = out['encoder_last_hidden_state'][:,:1,:]
        # out1 = torch.avg_pool1d(out1, kernel_size=out1.shape[-1]).squeeze(-1)
        # out2 = out['encoder_last_hidden_state'][:, 1:, :]
        out = self.clf(out['encoder_last_hidden_state'])
        # print(out.shape)
        # print(out1.shape)
        return out#bs 40 3855


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, 1024)
        self.act = act_layer()
        self.fc3 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    net = model().to(device)
    net(torch.zeros((10,3,96,96)).to(device),torch.zeros((10,96,96)).to(device))
    # for name, param in net.named_parameters():
        # param.requires_grad = False
        # print(name)
    # con1 = net.encoder.config
    # print(con1)

    print(count_parameters(net))