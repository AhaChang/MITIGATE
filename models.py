import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class Discriminator(nn.Module):
    def __init__(self, nin, nout, dropout):
        super(Discriminator, self).__init__()

        self.lr1 = nn.Linear(nin, nin//4)
        self.lr2 = nn.Linear(nin//4, nout)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lr1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr2(x)
        return x
    
    
class MultiTask(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MultiTask, self).__init__()

        self.encoder = Encoder(nfeat, nhid, dropout)
        self.dis_neighbor = Discriminator(nhid, nout, dropout)
        self.dis_anomaly = Discriminator(nhid, 2, dropout) 

    def forward(self, x, adj):
        x = self.encoder(x, adj)
        pred_nc = self.dis_neighbor(x)
        pred_ad = self.dis_anomaly(x)
        return x, pred_nc, pred_ad


class NCModel(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout) -> None:
        super(NCModel, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        embed = self.gc1(x, adj)
        x = F.dropout(F.relu(embed), self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return embed, x


class ADModel(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout) -> None:
        super(ADModel, self).__init__()

        self.encoder = Encoder(nfeat, nhid, dropout)
        self.discriminator = Discriminator(nhid, nout, dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        embed = self.encoder(x, adj)
        pred = self.discriminator(embed)
        return embed, pred


class SingleTask(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(SingleTask, self).__init__()

        self.encoder = Encoder(nfeat, nhid, dropout)
        self.discriminator = Discriminator(nhid, nout, dropout)

    def forward(self, x, adj):
        x = self.encoder(x, adj)
        pred = self.discriminator(x)
        return x, pred
    

class LinearSelector(nn.Module):
    def __init__(self, nfeat, nhid) -> None:
        super(LinearSelector, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, 2)
    
    def forward(self, feats):
        x = self.fc1(feats)
        x = self.fc2(F.relu(x))
        return x
