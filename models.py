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

        self.lr1 = nn.Linear(nin, 16)
        self.lr2 = nn.Linear(16, nout)
        self.dropout = dropout

    def forward(self, x):
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


class Model1(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(Model1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dis_anomaly = nn.Linear(nhid, 2) 
        self.dropout = dropout

    def forward(self, x, adj):
        embed = self.gc1(x, adj)
        x = F.dropout(F.relu(embed), self.dropout, training=self.training)
        pred_nc = self.gc2(x, adj)
        pred_ad = self.dis_anomaly(x)
        return embed, pred_nc, pred_ad


class Model2(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(Model2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nout)
        self.lr = nn.Linear(nfeat, 2) 
        self.dropout = dropout

    def forward(self, x, adj):
        pred_nc = self.gc1(x, adj)
        pred_ad = self.lr(x)
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

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dis_anomaly = nn.Linear(nhid, nout) 
        self.dropout = dropout

    def forward(self, x, adj):
        embed = self.gc1(x, adj)
        x = F.dropout(F.relu(embed), self.dropout, training=self.training)
        pred_ad = self.dis_anomaly(x)
        return embed, pred_ad


import layers

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.fc = nn.Linear(n_in, n_h)
        self.gcn = layers.GCN(n_in, n_h, activation)
        self.read = layers.AvgReadout()
        self.act = nn.PReLU()

        self.sigm = nn.Sigmoid()

        self.disc = layers.Discriminator(n_h)

    def forward(self, x_1, x_2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(x_1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(x_2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        # h_1 = self.sigm(self.fc(seq))
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()