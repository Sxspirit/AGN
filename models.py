import torch.nn as nn
import torch.nn.functional as F
import torch
from gcn.layers import GraphConvolution, MLPLayer
from Citation.gcn.layers import Attention

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)
        self.dropout = dropout
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        return x

class LAGCN1(torch.nn.Module):
    def __init__(self, concat, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LAGCN1, self).__init__()

        self.convs_initial = torch.nn.ModuleList()
        self.bns_initial = torch.nn.ModuleList()
        for _ in range(concat):
            self.convs_initial.append(GraphConvolution(in_channels, hidden_channels))
            self.bns_initial.append(torch.nn.BatchNorm1d(hidden_channels))
        self.linn = torch.nn.Linear(concat*hidden_channels+concat*hidden_channels,1)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.linn2 = torch.nn.Linear(concat*hidden_channels,out_channels)
        self.ml = MLPLayer(concat*hidden_channels,hidden_channels)
        for _ in range(num_layers - 2):
            self.convs.append(
                GraphConvolution(concat*hidden_channels, concat*hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(concat*hidden_channels))
        self.convs.append(GraphConvolution(concat*hidden_channels, hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs_initial:
            conv.reset_parameters()
        for bn in self.bns_initial:
            bn.reset_parameters()
        self.linn.reset_parameters()
        self.linn2.reset_parameters()

    def forward(self, x_list, adj_t):
        hidden_list = []
        for i, conv in enumerate(self.convs_initial):
            x = conv(x_list[i], adj_t)
            x = self.bns_initial[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)
        num_node = x.shape[0]
        input = x
        for i, conv in enumerate(self.convs[:-1]):
            alpha = torch.sigmoid(self.linn(torch.cat([input,x],dim=1))).view(num_node,1)
            x = alpha*x + (1-alpha)*input
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = alpha * x + (1 - alpha) * input
        # x = self.ml(x)
        x = self.convs[-1](x, adj_t)
        x = F.log_softmax(x,dim=1)

        return x


class SFGCN(nn.Module):
    def __init__(self, concat,nfeat,  nhid,nclass, numlayers,  dropout):
        super(SFGCN, self).__init__()

        # self.SGCN1 = gcn_air1(6,nfeat, nhid1, nhid2,dropout=dropout ,num_hops=6)
        # self.SGCN2 = gcn_air1(6,nfeat, nhid1, nhid2,dropout=dropout, num_hops=6)
        # self.CGCN = gcn_air1(6,nfeat, nhid1, nhid2, dropout=dropout,num_hops=6)
        self.SGCN1 = LAGCN1(concat,nfeat, nhid,nclass,numlayers,dropout=dropout )
        self.SGCN2 = LAGCN1(concat,nfeat, nhid,nclass,numlayers,dropout=dropout)
        self.CGCN = LAGCN1(concat,nfeat, nhid, nclass, numlayers,dropout=dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, adj,nsadj):
        emb1 = self.SGCN1(x, adj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, adj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, nsadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, nsadj) # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output, att, emb1, com1, com2, emb2, emb


