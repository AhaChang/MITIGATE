import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # # c_x shape (features)
        # # h_pl shape (nodes, features)
        #
        # c_x = torch.unsqueeze(c, 0)
        # # print(f'c_x.size(): {c_x.size()}')
        # # print(f'c_x: {c_x}')
        # c_x = c_x.expand_as(h_pl)
        # # print(f'c_x.size(): {c_x.size()}')
        # # print(f'c_x: {c_x}')
        # print(f'self.f_k(h_pl, c_x).size(): {self.f_k(h_pl, c_x).size()}')
        # sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        # sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        #
        # if s_bias1 is not None:
        #     sc_1 += s_bias1
        # if s_bias2 is not None:
        #     sc_2 += s_bias2
        #
        # logits = torch.cat((sc_1, sc_2), 0)
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


# Applies an average on seq, of shape (nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        # if msk is None:
        #     return torch.mean(seq, 0)
        # else:
        #     msk = torch.unsqueeze(msk, -1)
        #     return torch.sum(seq * msk, 0) / torch.sum(msk)
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)
